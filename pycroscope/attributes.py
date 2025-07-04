"""

Code for retrieving the value of attributes.

"""

import ast
import inspect
import sys
import types
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar, Optional, Union

from typing_extensions import assert_never

from .annotated_types import EnumName
from .annotations import Context, annotation_expr_from_annotations, type_from_runtime
from .maybe_asynq import asynq
from .options import Options, PyObjectSequenceOption
from .safe import is_bound_classmethod, safe_isinstance, safe_issubclass
from .signature import MaybeSignature
from .stacked_scopes import Composite
from .value import (
    UNINITIALIZED_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    CustomCheckExtension,
    GenericBases,
    GenericValue,
    HasAttrExtension,
    IntersectionValue,
    KnownValue,
    KnownValueWithTypeVars,
    MultiValuedValue,
    Qualifier,
    SubclassValue,
    SyntheticModuleValue,
    TypedValue,
    TypeVarValue,
    UnboundMethodValue,
    Value,
    annotate_value,
    replace_fallback,
    set_self,
)

# these don't appear to be in the standard types module
SlotWrapperType = type(type.__init__)
MethodDescriptorType = type(list.append)
NoneType = type(None)


@dataclass
class AttrContext:
    root_composite: Composite
    attr: str
    options: Options = field(repr=False)
    skip_mro: bool
    skip_unwrap: bool
    prefer_typeshed: bool

    @property
    def root_value(self) -> Value:
        return self.root_composite.value

    def record_usage(self, obj: Any, val: Value) -> None:
        pass

    def record_attr_read(self, obj: Any) -> None:
        pass

    def get_property_type_from_argspec(self, obj: property) -> Value:
        return AnyValue(AnySource.inference)

    def resolve_name_from_typeshed(self, module: str, name: str) -> Value:
        return UNINITIALIZED_VALUE

    def get_attribute_from_typeshed(self, typ: type, *, on_class: bool) -> Value:
        return UNINITIALIZED_VALUE

    def get_attribute_from_typeshed_recursively(
        self, fq_name: str, *, on_class: bool
    ) -> tuple[Value, object]:
        return UNINITIALIZED_VALUE, None

    def should_ignore_none_attributes(self) -> bool:
        return False

    def get_signature(self, obj: object) -> MaybeSignature:
        return None

    def get_generic_bases(
        self, typ: Union[type, str], generic_args: Sequence[Value]
    ) -> GenericBases:
        return {}


def get_attribute(ctx: AttrContext) -> Value:
    root_value = replace_fallback(ctx.root_value)
    if isinstance(root_value, KnownValue):
        attribute_value = _get_attribute_from_known(root_value.val, ctx)
    elif isinstance(root_value, TypedValue):
        if (
            isinstance(root_value, CallableValue)
            and ctx.attr == "asynq"
            and root_value.signature.is_asynq
        ):
            return root_value.get_asynq_value()
        if isinstance(root_value, GenericValue):
            args = root_value.args
        else:
            args = ()
        if isinstance(root_value.typ, str):
            attribute_value = _get_attribute_from_synthetic_type(
                root_value.typ, args, ctx
            )
        else:
            attribute_value = _get_attribute_from_typed(root_value.typ, args, ctx)
    elif isinstance(root_value, SubclassValue):
        if isinstance(root_value.typ, TypedValue):
            if isinstance(root_value.typ.typ, str):
                # TODO handle synthetic types
                return AnyValue(AnySource.inference)
            attribute_value = _get_attribute_from_subclass(
                root_value.typ.typ, root_value.typ, ctx
            )
        elif isinstance(root_value.typ, AnyValue):
            attribute_value = AnyValue(AnySource.from_another)
        else:
            attribute_value = _get_attribute_from_known(type, ctx)
    elif isinstance(root_value, UnboundMethodValue):
        attribute_value = _get_attribute_from_unbound(root_value, ctx)
    elif isinstance(root_value, AnyValue):
        attribute_value = AnyValue(AnySource.from_another)
    elif isinstance(root_value, MultiValuedValue):
        raise TypeError("caller should unwrap MultiValuedValue")
    elif isinstance(root_value, IntersectionValue):
        raise TypeError("caller should unwrap IntersectionValue")
    elif isinstance(root_value, SyntheticModuleValue):
        module = ".".join(root_value.module_path)
        attribute_value = ctx.resolve_name_from_typeshed(module, ctx.attr)
    else:
        assert_never(root_value)
    if (
        isinstance(attribute_value, AnyValue) or attribute_value is UNINITIALIZED_VALUE
    ) and isinstance(ctx.root_value, AnnotatedValue):
        for guard in ctx.root_value.get_metadata_of_type(HasAttrExtension):
            if guard.attribute_name == KnownValue(ctx.attr):
                return guard.attribute_type
    return attribute_value


def may_have_dynamic_attributes(typ: type) -> bool:
    """These types have typeshed stubs, but instances may have other attributes."""
    if typ is type or typ is super or typ is types.FunctionType:
        return True
    return False


def _get_attribute_from_subclass(
    typ: type, self_value: Value, ctx: AttrContext
) -> Value:
    ctx.record_attr_read(typ)

    # First check values that are special in Python
    if ctx.attr == "__class__":
        return KnownValue(type(typ))
    elif ctx.attr == "__dict__":
        return TypedValue(dict)
    elif ctx.attr == "__bases__":
        return GenericValue(tuple, [SubclassValue(TypedValue(object))])
    result, _, should_unwrap = _get_attribute_from_mro(typ, ctx, on_class=True)
    if should_unwrap:
        result = _unwrap_value_from_subclass(result, ctx)
    result = set_self(result, self_value)
    ctx.record_usage(typ, result)
    return result


_TCAA = Callable[[object], bool]


class TreatClassAttributeAsAny(PyObjectSequenceOption[_TCAA]):
    """Allows treating certain class attributes as Any.

    Instances of this option are callables that take an object found among
    a class's attributes and return True if the attribute should instead
    be treated as Any.

    """

    default_value: Sequence[_TCAA] = [
        lambda cls_val: cls_val is None or cls_val is NotImplemented
    ]
    name = "treat_class_attribute_as_any"

    @classmethod
    def should_treat_as_any(cls, val: object, options: Options) -> bool:
        option_value = options.get_value_for(cls)
        return any(func(val) for func in option_value)


_CAT = Callable[[object], Optional[tuple[Value, Value]]]


class ClassAttributeTransformer(PyObjectSequenceOption[_CAT]):
    """Transform certain class attributes.

    Instances of this option are callables that take an object found among
    a class's attributes and return either None (if the value should not
    be transformed) or a pair of a get and set type. To disallow setting
    the value, return :data:`pycroscope.value.NO_RETURN_VALUE`.

    If multiple transformers match an object, the first one is used.

    TODO: The set type is currently ignored.

    """

    default_value: ClassVar[Sequence[_CAT]] = []
    name = "class_attribute_transformers"

    @classmethod
    def transform_attribute(cls, val: object, options: Options) -> Optional[Value]:
        option_value = options.get_value_for(cls)
        for transformer in option_value:
            result = transformer(val)
            if result is not None:
                return result[0]
        return None


def _unwrap_value_from_subclass(result: Value, ctx: AttrContext) -> Value:
    if not isinstance(result, KnownValue) or ctx.skip_unwrap:
        return result
    cls_val = result.val
    if (
        isinstance(
            cls_val,
            (
                types.FunctionType,
                types.MethodType,
                MethodDescriptorType,
                SlotWrapperType,
                classmethod,
                staticmethod,
            ),
        )
        or (
            # non-static method
            _static_hasattr(cls_val, "decorator")
            and _static_hasattr(cls_val, "instance")
            and not isinstance(cls_val.instance, type)
        )
        or (asynq is not None and asynq.is_async_fn(cls_val))
    ):
        # static or class method
        return KnownValue(cls_val)
    elif _static_hasattr(cls_val, "__get__"):
        return AnyValue(AnySource.inference)  # can't figure out what this will return
    elif TreatClassAttributeAsAny.should_treat_as_any(cls_val, ctx.options):
        return AnyValue(AnySource.error)
    else:
        transformed = ClassAttributeTransformer.transform_attribute(
            cls_val, ctx.options
        )
        if transformed is not None:
            return transformed
        return KnownValue(cls_val)


def _get_attribute_from_synthetic_type(
    fq_name: str, generic_args: Sequence[Value], ctx: AttrContext
) -> Value:
    # First check values that are special in Python
    if ctx.attr == "__class__":
        # TODO: a KnownValue for synthetic types?
        return AnyValue(AnySource.inference)
    elif ctx.attr == "__dict__":
        return TypedValue(dict)
    result, provider = ctx.get_attribute_from_typeshed_recursively(
        fq_name, on_class=False
    )
    result = _substitute_typevars(fq_name, generic_args, result, provider, ctx)
    result = set_self(result, ctx.root_value)
    return result


def _get_attribute_from_typed(
    typ: type, generic_args: Sequence[Value], ctx: AttrContext
) -> Value:
    ctx.record_attr_read(typ)

    # First check values that are special in Python
    if ctx.attr == "__class__":
        return KnownValue(typ)
    elif ctx.attr == "__dict__":
        return TypedValue(dict)

    result, provider, should_unwrap = _get_attribute_from_mro(typ, ctx, on_class=False)
    result = _substitute_typevars(typ, generic_args, result, provider, ctx)
    if should_unwrap:
        result = _unwrap_value_from_typed(result, typ, ctx)
    ctx.record_usage(typ, result)
    result = set_self(result, ctx.root_value)
    if ctx.attr == "name" and safe_issubclass(typ, Enum) and result == TypedValue(str):
        return annotate_value(result, [CustomCheckExtension(EnumName(typ))])
    return result


def _substitute_typevars(
    typ: Union[type, str],
    generic_args: Sequence[Value],
    result: Value,
    provider: object,
    ctx: AttrContext,
) -> Value:
    if isinstance(typ, (type, str)):
        generic_bases = ctx.get_generic_bases(typ, generic_args)
    else:
        generic_bases = {}
    if provider in generic_bases:
        result = result.substitute_typevars(generic_bases[provider])
    if generic_args and typ in generic_bases:
        typevars = [
            val.typevar
            for val in generic_bases[typ].values()
            if isinstance(val, TypeVarValue)
        ]
        tv_map = dict(zip(typevars, generic_args))
        result = result.substitute_typevars(tv_map)
    return result


def _unwrap_value_from_typed(result: Value, typ: type, ctx: AttrContext) -> Value:
    if not isinstance(result, KnownValue) or ctx.skip_unwrap:
        return result
    typevars = result.typevars if isinstance(result, KnownValueWithTypeVars) else None
    cls_val = result.val
    if isinstance(cls_val, property):
        return ctx.get_property_type_from_argspec(cls_val)
    elif is_bound_classmethod(cls_val):
        return result
    elif inspect.isfunction(cls_val):
        # either a staticmethod or an unbound method
        try:
            descriptor = inspect.getattr_static(typ, ctx.attr)
        except AttributeError:
            # probably a super call; assume unbound method
            if ctx.attr != "__new__":
                return UnboundMethodValue(
                    ctx.attr, ctx.root_composite, typevars=typevars
                )
            else:
                # __new__ is implicitly a staticmethod
                return result
        if isinstance(descriptor, staticmethod) or ctx.attr == "__new__":
            return result
        else:
            return UnboundMethodValue(ctx.attr, ctx.root_composite, typevars=typevars)
    elif isinstance(cls_val, (types.MethodType, MethodDescriptorType, SlotWrapperType)):
        # built-in method; e.g. scope_lib.tests.SimpleDatabox.get
        return UnboundMethodValue(ctx.attr, ctx.root_composite, typevars=typevars)
    elif (
        _static_hasattr(cls_val, "decorator")
        and _static_hasattr(cls_val, "instance")
        and not isinstance(cls_val.instance, type)
    ):
        # non-static method
        return UnboundMethodValue(ctx.attr, ctx.root_composite, typevars=typevars)
    elif asynq is not None and asynq.is_async_fn(cls_val):
        # static or class method
        return result
    elif _static_hasattr(cls_val, "func_code"):
        # Cython function probably
        return UnboundMethodValue(ctx.attr, ctx.root_composite, typevars=typevars)
    elif _static_hasattr(cls_val, "__get__"):
        typeshed_type = ctx.get_attribute_from_typeshed(typ, on_class=False)
        if typeshed_type is not UNINITIALIZED_VALUE:
            return typeshed_type
        return AnyValue(AnySource.inference)
    elif TreatClassAttributeAsAny.should_treat_as_any(cls_val, ctx.options):
        return AnyValue(AnySource.error)
    else:
        transformed = ClassAttributeTransformer.transform_attribute(
            cls_val, ctx.options
        )
        if transformed is not None:
            return transformed
        return result


_KAH = Callable[[object, str], Optional[Value]]


def _default_transformer(obj: object, attr: str) -> Optional[Value]:
    # Type alias to Any
    if obj is Any:
        return AnyValue(AnySource.explicit)

    # Avoid generating huge Union type with the actual value
    if obj is sys and attr == "modules":
        return GenericValue(dict, [TypedValue(str), TypedValue(types.ModuleType)])

    return None


class KnownAttributeHook(PyObjectSequenceOption[_KAH]):
    """Allows hooking into the inferred value for an attribute on a literal."""

    default_value: ClassVar[Sequence[_KAH]] = [_default_transformer]
    name = "known_attribute_hook"

    @classmethod
    def get_attribute(cls, obj: object, attr: str, options: Options) -> Optional[Value]:
        option_value = options.get_value_for(cls)
        for transformer in option_value:
            result = transformer(obj, attr)
            if result is not None:
                return result
        return None


def _get_attribute_from_known(obj: object, ctx: AttrContext) -> Value:
    ctx.record_attr_read(type(obj))

    if (obj is None or obj is NoneType) and ctx.should_ignore_none_attributes():
        # This usually indicates some context is set to None
        # in the module and initialized later.
        return AnyValue(AnySource.error)

    hooked_value = KnownAttributeHook.get_attribute(obj, ctx.attr, ctx.options)
    if hooked_value is not None:
        return hooked_value

    result, _, _ = _get_attribute_from_mro(obj, ctx, on_class=True)
    if (
        isinstance(result, KnownValue)
        and (
            safe_isinstance(result.val, types.MethodType)
            or safe_isinstance(result.val, types.BuiltinFunctionType)
            and result.val.__self__ is obj
        )
        and isinstance(ctx.root_value, AnnotatedValue)
    ):
        result = set_self(result, ctx.root_value)
    elif safe_isinstance(obj, type):
        result = set_self(result, TypedValue(obj))
    if isinstance(obj, (types.ModuleType, type)):
        ctx.record_usage(obj, result)
    else:
        ctx.record_usage(type(obj), result)
    return result


def _get_attribute_from_unbound(
    root_value: UnboundMethodValue, ctx: AttrContext
) -> Value:
    if root_value.secondary_attr_name is not None:
        return AnyValue(AnySource.inference)
    method = root_value.get_method()
    if method is None:
        return AnyValue(AnySource.inference)
    try:
        getattr(method, ctx.attr)
    except AttributeError:
        return UNINITIALIZED_VALUE
    result = UnboundMethodValue(
        root_value.attr_name, root_value.composite, secondary_attr_name=ctx.attr
    )
    ctx.record_usage(type(method), result)
    return result


@dataclass
class AnnotationsContext(Context):
    attr_ctx: AttrContext
    cls: object

    def __post_init__(self) -> None:
        super().__init__()

    def get_name(self, node: ast.Name) -> Value:
        try:
            if isinstance(self.cls, types.ModuleType):
                globals = self.cls.__dict__
            else:
                globals = sys.modules[self.cls.__module__].__dict__
        except Exception:
            return AnyValue(AnySource.error)
        else:
            return self.get_name_from_globals(node.id, globals)

    def get_signature(self, callable: object) -> MaybeSignature:
        return self.attr_ctx.get_signature(callable)


def _get_triple_from_annotations(
    annotations: dict[str, object], typ: object, ctx: AttrContext
) -> Optional[tuple[Value, object, bool]]:
    attr_expr = annotation_expr_from_annotations(
        annotations, ctx.attr, ctx=AnnotationsContext(ctx, typ)
    )
    if attr_expr is not None:
        attr_type, _ = attr_expr.maybe_unqualify({Qualifier.ClassVar, Qualifier.Final})
        if attr_type is not None:
            return (attr_type, typ, False)
    return None


def _get_attribute_from_mro(
    typ: object, ctx: AttrContext, on_class: bool
) -> tuple[Value, object, bool]:
    # Then go through the MRO and find base classes that may define the attribute.
    if safe_isinstance(typ, type) and safe_issubclass(typ, Enum):
        # Special case, to avoid picking an attribute of Enum instances (e.g., name)
        # over an Enum member. Ideally we'd have a more principled way to support this
        # but I haven't thought of one.
        try:
            return KnownValue(getattr(typ, ctx.attr)), typ, True
        except Exception:
            pass
    elif safe_isinstance(typ, types.ModuleType):
        try:
            annotations = typ.__annotations__
        except Exception:
            pass
        else:
            triple = _get_triple_from_annotations(annotations, typ, ctx)
            if triple is not None:
                return triple

    if safe_isinstance(typ, type):
        try:
            mro = list(type.mro(typ))
        except Exception:
            # broken mro method
            pass
        else:
            for base_cls in mro:
                if ctx.skip_mro and base_cls is not typ:
                    continue

                typeshed_type = ctx.get_attribute_from_typeshed(
                    base_cls, on_class=on_class or ctx.skip_unwrap
                )
                if typeshed_type is not UNINITIALIZED_VALUE:
                    if ctx.prefer_typeshed:
                        return typeshed_type, base_cls, False
                    # If it's a callable, we'll probably do better
                    # getting the attribute from the type ourselves,
                    # because we may have our own implementation.
                    if not isinstance(typeshed_type, CallableValue):
                        return typeshed_type, base_cls, False

                try:
                    base_dict = base_cls.__dict__
                except Exception:
                    continue

                try:
                    # Make sure to use only __annotations__ that are actually on this
                    # class, not ones inherited from a base class.
                    # Starting in 3.10, __annotations__ is not inherited.
                    if sys.version_info >= (3, 10):
                        annotations = base_cls.__annotations__
                    else:
                        annotations = base_dict["__annotations__"]
                except Exception:
                    pass
                else:
                    triple = _get_triple_from_annotations(annotations, base_cls, ctx)
                    if triple is not None:
                        return triple

                try:
                    # Make sure we use only the object from this class, but do invoke
                    # the descriptor protocol with getattr.
                    base_dict[ctx.attr]
                except Exception:
                    pass
                else:
                    try:
                        val = KnownValue(getattr(typ, ctx.attr))
                    except Exception:
                        val = AnyValue(AnySource.inference)
                    return val, base_cls, True

                if typeshed_type is not UNINITIALIZED_VALUE:
                    return typeshed_type, base_cls, False

    attrs_type = get_attrs_attribute(typ, ctx)
    if attrs_type is not None:
        return attrs_type, typ, False

    if not ctx.skip_mro:
        # Even if we didn't find it any __dict__, maybe getattr() finds it directly.
        try:
            return KnownValue(getattr(typ, ctx.attr)), typ, True
        except AttributeError:
            pass
        except Exception:
            # It exists, but has a broken __getattr__ or something
            return AnyValue(AnySource.inference), typ, True

    return UNINITIALIZED_VALUE, object, False


def _static_hasattr(value: object, attr: str) -> bool:
    """Returns whether this value has the given attribute, ignoring __getattr__ overrides."""
    try:
        object.__getattribute__(value, attr)
    except AttributeError:
        return False
    else:
        return True


def get_attrs_attribute(typ: object, ctx: AttrContext) -> Optional[Value]:
    try:
        if hasattr(typ, "__attrs_attrs__"):
            for attr_attr in typ.__attrs_attrs__:
                if attr_attr.name == ctx.attr:
                    if attr_attr.type is not None:
                        return type_from_runtime(
                            attr_attr.type, ctx=AnnotationsContext(ctx, typ)
                        )
                    else:
                        return AnyValue(AnySource.unannotated)
    except Exception:
        # Guard against silly objects throwing exceptions on hasattr()
        # or similar shenanigans.
        pass
    return None
