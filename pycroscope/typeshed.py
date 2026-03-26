"""

Code for getting annotations from typeshed (and from third-party stubs generally).

"""

import ast
import builtins
import collections.abc
import inspect
import sys
import types
from abc import abstractmethod
from collections.abc import Collection, Iterable, MutableMapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from enum import EnumMeta
from functools import lru_cache
from pathlib import Path
from types import GeneratorType, MethodDescriptorType, ModuleType
from typing import Any, Generic, TypeVar

import typeshed_client
from typing_extensions import Protocol

from .analysis_lib import is_positional_only_arg_name
from .annotations import (
    Context,
    DecoratorValue,
    SyntheticEvaluator,
    annotation_expr_from_value,
    value_from_ast,
)
from .error_code import Error, ErrorCode
from .extensions import deprecated as deprecated_decorator
from .extensions import evaluated, overload, real_overload
from .functions import translate_vararg_type
from .input_sig import InputSigValue
from .options import (
    InvalidConfigOption,
    OptionalPathOption,
    Options,
    PathSequenceOption,
)
from .safe import hasattr_static, is_typing_name, safe_isinstance
from .shared_options import ImportPaths
from .signature import (
    ConcreteSignature,
    OverloadedSignature,
    ParameterKind,
    Signature,
    SigParameter,
    make_bound_method,
)
from .stacked_scopes import Composite, uniq_chain
from .value import (
    UNINITIALIZED_VALUE,
    AnnotationExpr,
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignContext,
    ClassSymbol,
    DeprecatedExtension,
    Extension,
    GenericValue,
    KnownValue,
    PropertyInfo,
    Qualifier,
    SubclassValue,
    SyntheticClassObjectValue,
    SyntheticModuleValue,
    TypedDictEntry,
    TypedDictValue,
    TypedValue,
    TypeParam,
    TypeVarParam,
    TypeVarTupleParam,
    TypeVarTupleValue,
    TypeVarValue,
    UninitializedValue,
    Value,
    annotate_value,
    iter_type_params_in_value,
    make_coro_type,
    replace_fallback,
    type_param_to_value,
    unannotate_value,
    unite_values,
)

PROPERTY_LIKE = {KnownValue(property), KnownValue(types.DynamicClassAttribute)}
_CLASS_SYMBOL_ALLOWED_QUALIFIERS = frozenset(
    {Qualifier.ClassVar, Qualifier.Final, Qualifier.ReadOnly, Qualifier.InitVar}
)

if sys.version_info >= (3, 11):
    from enum import property as enum_property

    PROPERTY_LIKE.add(KnownValue(enum_property))


T_co = TypeVar("T_co", covariant=True)


@lru_cache(maxsize=1)
def _get_default_typeshed_search_context() -> Any:
    return typeshed_client.get_search_context()


def _resolve_typeshed_path(typeshed_path: Path | None) -> Path | None:
    if typeshed_path is None:
        return None
    if (typeshed_path / "VERSIONS").is_file():
        return typeshed_path
    stdlib_path = typeshed_path / "stdlib"
    if (stdlib_path / "VERSIONS").is_file():
        return stdlib_path
    raise InvalidConfigOption(
        "Invalid value for option typeshed_path: expected a path containing VERSIONS "
        f"or stdlib/VERSIONS, but got {typeshed_path}"
    )


@lru_cache(maxsize=10)
def _get_resolver_for_stub_paths(
    extra_paths: tuple[str, ...], typeshed_path: str | None
) -> typeshed_client.Resolver:
    default_ctx = _get_default_typeshed_search_context()
    search_path = [*default_ctx.search_path, *(Path(path) for path in extra_paths)]
    typeshed = default_ctx.typeshed if typeshed_path is None else Path(typeshed_path)
    ctx = typeshed_client.get_search_context(
        typeshed=typeshed,
        search_path=search_path,
        version=default_ctx.version,
        platform=default_ctx.platform,
        raise_on_warnings=default_ctx.raise_on_warnings,
        allow_py_files=default_ctx.allow_py_files,
    )
    return typeshed_client.Resolver(ctx)


@dataclass
class _AnnotationContext(Context):
    finder: "TypeshedFinder"
    module: str

    def show_error(
        self,
        message: str,
        error_code: Error = ErrorCode.invalid_annotation,
        node: ast.AST | None = None,
    ) -> None:
        # TODO: Make this error, at least in tests, so we know about missing support
        # print(f"Error in annotation: {message}")
        self.finder.log(message, ())

    def get_name(self, node: ast.Name) -> Value:
        return self.finder.resolve_name(self.module, node.id)

    def get_attribute(self, root_value: Value, node: ast.Attribute) -> Value:
        if isinstance(root_value, KnownValue):
            if isinstance(root_value.val, ModuleType):
                return self.finder.resolve_name(root_value.val.__name__, node.attr)
        elif isinstance(root_value, SyntheticModuleValue):
            return self.finder.resolve_name(".".join(root_value.module_path), node.attr)
        return super().get_attribute(root_value, node)


class StubPath(PathSequenceOption):
    """Extra paths in which to look for stubs."""

    name = "stub_path"


class TypeshedPath(OptionalPathOption):
    """Path to a typeshed checkout to use instead of typeshed_client's bundled copy.

    The path may either contain `VERSIONS` directly or have a `stdlib/` subdirectory
    containing `VERSIONS`.

    """

    name = "typeshed_path"
    is_global = True


# These are specified as just "List = _Alias()" in typing.pyi. Redirect
# them to the proper runtime equivalent.
_TYPING_ALIASES = {
    "typing.List": "builtins.list",
    "typing.Dict": "builtins.dict",
    "typing.DefaultDict": "collections.defaultdict",
    "typing.Set": "builtins.set",
    "typing.Frozenzet": "builtins.frozenset",
    "typing.Counter": "collections.Counter",
    "typing.Deque": "collections.deque",
    "typing.ChainMap": "collections.ChainMap",
    "typing.OrderedDict": "collections.OrderedDict",
    "typing.Tuple": "builtins.tuple",
}


@dataclass
class TypeshedFinder:
    ctx: CanAssignContext = field(repr=False)
    verbose: bool = True
    resolver: typeshed_client.Resolver = field(default_factory=typeshed_client.Resolver)
    _assignment_cache: dict[tuple[str, ast.AST], Value] = field(
        default_factory=dict, repr=False, init=False
    )
    _attribute_cache: dict[tuple[str, str, bool], Value] = field(
        default_factory=dict, repr=False, init=False
    )
    _direct_symbol_cache: dict[tuple[str, str], ClassSymbol | None] = field(
        default_factory=dict, repr=False, init=False
    )
    _active_infos: list[typeshed_client.resolver.ResolvedName] = field(
        default_factory=list, repr=False, init=False
    )
    _info_cache: dict[str, typeshed_client.resolver.ResolvedName] = field(
        default_factory=dict, repr=False, init=False
    )

    @classmethod
    def make(
        cls,
        can_assign_ctx: CanAssignContext,
        options: Options,
        *,
        verbose: bool = False,
    ) -> "TypeshedFinder":
        extra_paths = tuple(
            str(path)
            for path in (
                Path.cwd(),
                *options.get_value_for(StubPath),
                *options.get_value_for(ImportPaths),
            )
        )
        typeshed = _resolve_typeshed_path(options.get_value_for(TypeshedPath))
        typeshed_path = None if typeshed is None else str(typeshed)
        resolver = _get_resolver_for_stub_paths(extra_paths, typeshed_path)
        return TypeshedFinder(can_assign_ctx, verbose, resolver)

    def log(self, message: str, obj: object) -> None:
        if not self.verbose:
            return
        print(f"{message}: {obj!r}")

    def _get_sig_from_method_descriptor(
        self, obj: MethodDescriptorType, allow_call: bool
    ) -> ConcreteSignature | None:
        objclass = obj.__objclass__
        fq_name = self._get_fq_name(objclass)
        if fq_name is None:
            return None
        info = self._get_info_for_name(fq_name)
        sig = self._get_method_signature_from_info(
            info, obj, fq_name, objclass.__module__, objclass, allow_call=allow_call
        )
        return sig

    def get_argspec(
        self,
        obj: object,
        *,
        allow_call: bool = False,
        type_params: Sequence[TypeParam] = (),
    ) -> ConcreteSignature | None:
        if isinstance(obj, str):
            # Synthetic type
            return self.get_argspec_for_fully_qualified_name(
                obj, obj, type_params=type_params
            )
        if inspect.ismethoddescriptor(obj) and hasattr_static(obj, "__objclass__"):
            return self._get_sig_from_method_descriptor(obj, allow_call)
        if inspect.isbuiltin(obj) and isinstance(obj.__self__, type):
            # This covers cases like dict.fromkeys and type.__subclasses__. We
            # want to make sure we get the underlying method descriptor object,
            # which we can apparently only get out of the __dict__.
            method = obj.__self__.__dict__.get(obj.__name__)
            if (
                method is not None
                and inspect.ismethoddescriptor(method)
                and hasattr_static(method, "__objclass__")
            ):
                sig = self._get_sig_from_method_descriptor(method, allow_call)
                if sig is None:
                    return None
                bound = make_bound_method(
                    sig, Composite(TypedValue(obj.__self__)), ctx=self.ctx
                )
                if bound is None:
                    return None
                return bound.get_signature(ctx=self.ctx)

        if inspect.ismethod(obj):
            self.log("Ignoring method", obj)
            return None
        if (
            hasattr_static(obj, "__qualname__")
            and hasattr_static(obj, "__name__")
            and hasattr_static(obj, "__module__")
            and isinstance(obj.__qualname__, str)
            and obj.__qualname__ != obj.__name__
            and "." in obj.__qualname__
        ):
            parent_name, own_name = obj.__qualname__.rsplit(".", maxsplit=1)
            # Work around the stub using the wrong name.
            # TODO we should be able to resolve this anyway.
            if parent_name == "EnumType" and obj.__module__ == "enum":
                parent_fqn = "enum.EnumMeta"
            else:
                parent_fqn = f"{obj.__module__}.{parent_name}"
            parent_info = self._get_info_for_name(parent_fqn)
            if parent_info is not None:
                maybe_info = self._get_child_info(parent_info, own_name, obj.__module__)
                if maybe_info is not None:
                    info, mod = maybe_info
                    fq_name = f"{parent_fqn}.{own_name}"
                    sig = self._get_signature_from_info(
                        info, obj, fq_name, mod, allow_call=allow_call
                    )
                    return sig

        fq_name = self._get_fq_name(obj)
        if fq_name is None:
            return None
        return self.get_argspec_for_fully_qualified_name(
            fq_name, obj, allow_call=allow_call, type_params=type_params
        )

    def get_argspec_for_fully_qualified_name(
        self,
        fq_name: str,
        obj: object,
        *,
        allow_call: bool = False,
        type_params: Sequence[TypeParam] = (),
    ) -> ConcreteSignature | None:
        info = self._get_info_for_name(fq_name)
        mod, _ = fq_name.rsplit(".", maxsplit=1)
        sig = self._get_signature_from_info(
            info, obj, fq_name, mod, allow_call=allow_call, type_params=type_params
        )
        return sig

    def is_final(self, fq_name: str | type) -> bool:
        """Return whether this type is marked as final in the stubs."""
        if isinstance(fq_name, type):
            maybe_fq_name = self._get_fq_name(fq_name)
            if maybe_fq_name is None:
                return False
            fq_name = maybe_fq_name
        info = self._get_info_for_name(fq_name)
        mod, _ = fq_name.rsplit(".", maxsplit=1)
        return self._is_final_from_info(info, mod)

    def is_final_attribute(self, class_fq_name: str, attr: str) -> bool:
        """Return whether an attribute is marked as final in stubs."""
        info = self._get_info_for_name(class_fq_name)
        mod, _ = class_fq_name.rsplit(".", maxsplit=1)
        return self._is_final_attribute_from_info(info, mod, attr)

    def _is_final_from_info(
        self, info: typeshed_client.resolver.ResolvedName, mod: str
    ) -> bool:
        if info is None:
            return False
        if isinstance(info, typeshed_client.ImportedInfo):
            return self._is_final_from_info(info.info, mod)
        if isinstance(info, typeshed_client.NameInfo) and isinstance(
            info.ast, ast.ClassDef
        ):
            for deco in info.ast.decorator_list:
                deco_value = self._parse_expr(deco, mod)
                if isinstance(deco_value, KnownValue) and is_typing_name(
                    deco_value.val, "final"
                ):
                    return True
        return False

    def _is_final_attribute_from_info(
        self, info: typeshed_client.resolver.ResolvedName, mod: str, attr: str
    ) -> bool:
        if info is None:
            return False
        if isinstance(info, typeshed_client.ImportedInfo):
            return self._is_final_attribute_from_info(info.info, mod, attr)
        if not (
            isinstance(info, typeshed_client.NameInfo)
            and isinstance(info.ast, ast.ClassDef)
        ):
            return False
        for statement in info.ast.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if statement.name != attr:
                    continue
                for deco in statement.decorator_list:
                    deco_value = self._parse_expr(deco, mod)
                    if isinstance(deco_value, KnownValue) and is_typing_name(
                        deco_value.val, "final"
                    ):
                        return True
                continue
            if not (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == attr
            ):
                continue
            try:
                expr = self._parse_annotation(statement.annotation, mod)
            except Exception:
                continue
            _, qualifiers = expr.maybe_unqualify({Qualifier.Final})
            if Qualifier.Final in qualifiers:
                return True
        return False

    def get_bases(self, typ: type | str) -> list[Value] | None:
        """Return the base classes for this type, including generic bases."""
        assert isinstance(typ, str) or isinstance(typ, type), repr(typ)
        return self.get_bases_for_value(TypedValue(typ))

    def get_bases_for_value(self, val: Value) -> list[Value] | None:
        if isinstance(val, TypedValue):
            if isinstance(val.typ, type):
                typ = val.typ
                # The way AbstractSet/Set is handled between collections and typing is
                # too confusing, just hardcode it.
                if typ is AbstractSet:
                    return [
                        GenericValue(Collection, (TypeVarValue(TypeVarParam(T_co)),))
                    ]
                if typ is collections.abc.Callable:
                    return None
                if sys.version_info >= (3, 10) and typ is types.UnionType:
                    return None
                # In 3.11 it's named EnumType and EnumMeta is an alias, but the
                # stubs have it the other way around. We can't deal with that for now.
                if typ is EnumMeta:
                    return [TypedValue(type)]
                fq_name = self._get_fq_name(typ)
                if fq_name is None:
                    return None
            else:
                fq_name = val.typ
                if fq_name == "collections.abc.Set":
                    return [
                        GenericValue(Collection, (TypeVarValue(TypeVarParam(T_co)),))
                    ]
                elif fq_name == "contextlib.AbstractContextManager":
                    return [GenericValue(Protocol, (TypeVarValue(TypeVarParam(T_co)),))]
                elif fq_name in (
                    "typing.Callable",
                    "collections.abc.Callable",
                    "typing.Union",
                    "types.UnionType",
                ):
                    return None
                elif is_typing_name(fq_name, "TypedDict"):
                    return [
                        GenericValue(
                            MutableMapping, [TypedValue(str), TypedValue(object)]
                        )
                    ]
            return self.get_bases_for_fq_name(fq_name)
        return None

    def is_protocol(self, typ: type) -> bool:
        """Return whether this type is marked as a Protocol in the stubs."""
        fq_name = self._get_fq_name(typ)
        if fq_name is None:
            return False
        bases = self.get_bases_for_value(TypedValue(fq_name))
        if bases is None:
            return False
        return any(
            isinstance(base, TypedValue) and is_typing_name(base.typ, "Protocol")
            for base in bases
        )

    def get_bases_recursively(self, typ: type | str) -> list[Value]:
        stack = [TypedValue(typ)]
        seen = set()
        bases = []
        # TODO return MRO order
        while stack:
            next_base = stack.pop()
            if next_base in seen:
                continue
            seen.add(next_base)
            bases.append(next_base)
            new_bases = self.get_bases_for_value(next_base)
            if new_bases is not None:
                bases += new_bases
        return bases

    def get_bases_for_fq_name(self, fq_name: str) -> list[Value] | None:
        if fq_name in (
            "typing.Generic",
            "typing.Protocol",
            "typing_extensions.Protocol",
        ):
            return []
        info = self._get_info_for_name(fq_name)
        mod, _ = fq_name.rsplit(".", maxsplit=1)
        return self._get_bases_from_info(info, mod, fq_name)

    def get_attribute(self, typ: type, attr: str, *, on_class: bool) -> Value:
        """Return the stub for this attribute.

        Does not look at parent classes. Returns UNINITIALIZED_VALUE if no
        stub can be found.

        """
        fq_name = self._get_fq_name(typ)
        if fq_name is None:
            return UNINITIALIZED_VALUE
        return self.get_attribute_for_fq_name(fq_name, attr, on_class=on_class)

    def get_attribute_for_fq_name(
        self, fq_name: str, attr: str, *, on_class: bool
    ) -> Value:
        key = (fq_name, attr, on_class)
        try:
            return self._attribute_cache[key]
        except KeyError:
            info = self._get_info_for_name(fq_name)
            mod, _ = fq_name.rsplit(".", maxsplit=1)
            val = self._get_attribute_from_info(info, mod, attr, on_class=on_class)
            if isinstance(val, AnnotationExpr):
                val, _ = val.unqualify()
            self._attribute_cache[key] = val
            return val

    def get_direct_symbol(self, typ: type | str, attr: str) -> ClassSymbol | None:
        """Return the symbol declared directly on this class in stubs."""
        if isinstance(typ, str):
            fq_name = typ
        else:
            fq_name = self._get_fq_name(typ)
            if fq_name is None:
                return None
        key = (fq_name, attr)
        try:
            return self._direct_symbol_cache[key]
        except KeyError:
            info = self._get_info_for_name(fq_name)
            mod, _ = fq_name.rsplit(".", maxsplit=1)
            symbol = self._get_direct_symbol_from_info(info, mod, attr)
            self._direct_symbol_cache[key] = symbol
            return symbol

    def get_attribute_recursively(
        self, fq_name: str, attr: str, *, on_class: bool
    ) -> tuple[Value, type | str | None]:
        """Get an attribute from a fully qualified class.

        Returns a tuple (value, provider).

        """
        for base in self.get_bases_recursively(fq_name):
            if isinstance(base, TypedValue):
                if isinstance(base.typ, str):
                    possible_value = self.get_attribute_for_fq_name(
                        base.typ, attr, on_class=on_class
                    )
                else:
                    possible_value = self.get_attribute(
                        base.typ, attr, on_class=on_class
                    )
                if possible_value is not UNINITIALIZED_VALUE:
                    return possible_value, base.typ
        return UNINITIALIZED_VALUE, None

    def has_attribute(self, typ: type | str, attr: str) -> bool:
        """Whether this type has this attribute in the stubs.

        Also looks at base classes.

        """
        if self._has_own_attribute(typ, attr):
            return True
        bases = self.get_bases_for_value(TypedValue(typ))
        if bases is not None:
            for base in bases:
                if not isinstance(base, TypedValue):
                    continue
                typ = base.typ
                if typ is Generic or is_typing_name(typ, "Protocol"):
                    continue
                if self.has_attribute(base.typ, attr):
                    return True
        return False

    def get_all_attributes(self, typ: type | str) -> set[str]:
        if isinstance(typ, str):
            fq_name = typ
        else:
            fq_name = self._get_fq_name(typ)
            if fq_name is None:
                return set()
        info = self._get_info_for_name(fq_name)
        mod, _ = fq_name.rsplit(".", maxsplit=1)
        return self._get_all_attributes_from_info(info, mod)

    def has_stubs(self, typ: type | str) -> bool:
        if isinstance(typ, str):
            fq_name = typ
        else:
            fq_name = self._get_fq_name(typ)
            if fq_name is None:
                return False
        info = self._get_info_for_name(fq_name)
        return info is not None

    def resolve_name(self, module: str, name: str) -> Value:
        info = self._get_info_for_name(f"{module}.{name}")
        if info is not None:
            return self._value_from_info(info, module)
        elif hasattr(builtins, name):
            val = getattr(builtins, name)
            if val is None or isinstance(val, type):
                return KnownValue(val)
        # TODO change to UNINITIALIZED_VALUE
        return AnyValue(AnySource.inference)

    def resolve_name_if_present(self, module: str, name: str) -> Value | None:
        info = self._get_info_for_name(f"{module}.{name}")
        if info is None:
            return None
        return self._value_from_info(info, module)

    def _get_attribute_from_info(
        self,
        info: typeshed_client.resolver.ResolvedName,
        mod: str,
        attr: str,
        *,
        on_class: bool,
    ) -> Value | AnnotationExpr:
        if info is None:
            return UNINITIALIZED_VALUE
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_attribute_from_info(
                info.info, ".".join(info.source_module), attr, on_class=on_class
            )
        elif isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, ast.ClassDef):
                if info.child_nodes and attr in info.child_nodes:
                    child_info = info.child_nodes[attr]
                    if isinstance(child_info, typeshed_client.NameInfo):
                        return self._get_value_from_child_info(
                            child_info.ast,
                            mod,
                            on_class=on_class,
                            parent_name=info.ast.name,
                        )
                    assert False, repr(child_info)
                return UNINITIALIZED_VALUE
            elif isinstance(info.ast, ast.Assign):
                val = self._parse_type(info.ast.value, mod)
                if isinstance(val, KnownValue) and isinstance(val.val, type):
                    return self.get_attribute(val.val, attr, on_class=on_class)
                else:
                    return UNINITIALIZED_VALUE
            else:
                return UNINITIALIZED_VALUE
        return UNINITIALIZED_VALUE

    def _get_value_from_child_info(
        self,
        node: ast.AST | typeshed_client.OverloadedName | typeshed_client.ImportedName,
        mod: str,
        *,
        on_class: bool,
        parent_name: str,
    ) -> Value | AnnotationExpr:
        if isinstance(node, ast.AnnAssign):
            return self._parse_annotation(node.annotation, mod)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_property = False
            for decorator_node in node.decorator_list:
                decorator_value = self._parse_expr(decorator_node, mod)
                if decorator_value in PROPERTY_LIKE:
                    is_property = True
            if is_property:
                if node.returns:
                    return self._parse_type(node.returns, mod)
                else:
                    return AnyValue(AnySource.unannotated)
            else:
                # TODO: apply decorators to the return value
                sig = self._get_signature_from_func_def(
                    node, None, mod, autobind=not on_class, bind_classmethod=on_class
                )
                if sig is None:
                    return AnyValue(AnySource.inference)
                else:
                    return CallableValue(sig)
        elif isinstance(node, ast.ClassDef):
            # Should be a synthetic singleton class object, but class-valued
            # members in stubs are still modeled imprecisely.
            return AnyValue(AnySource.inference)
        elif isinstance(node, ast.Assign):
            return UNINITIALIZED_VALUE
        elif isinstance(node, typeshed_client.OverloadedName):
            sigs = []
            for subnode in node.definitions:
                val = self._get_value_from_child_info(
                    subnode, mod, on_class=on_class, parent_name=parent_name
                )
                if isinstance(val, AnnotationExpr):
                    val, _ = val.unqualify()
                sig = self._sig_from_value(val)
                if not isinstance(sig, Signature):
                    return AnyValue(AnySource.inference)
                sigs.append(sig)
            return CallableValue(OverloadedSignature(sigs))
        assert False, repr(node)

    def _get_child_info(
        self, info: typeshed_client.resolver.ResolvedName, attr: str, mod: str
    ) -> tuple[typeshed_client.resolver.ResolvedName, str] | None:
        if info is None:
            return None
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_child_info(info.info, attr, ".".join(info.source_module))
        elif isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, ast.ClassDef):
                if info.child_nodes and attr in info.child_nodes:
                    return info.child_nodes[attr], mod
                return None
            return None  # TODO maybe we need this for aliased methods
        return None

    def _get_direct_symbol_from_info(
        self, info: typeshed_client.resolver.ResolvedName, mod: str, attr: str
    ) -> ClassSymbol | None:
        if info is None:
            return None
        if isinstance(info, typeshed_client.ImportedInfo):
            return self._get_direct_symbol_from_info(
                info.info, ".".join(info.source_module), attr
            )
        if not (
            isinstance(info, typeshed_client.NameInfo)
            and isinstance(info.ast, ast.ClassDef)
            and info.child_nodes
            and attr in info.child_nodes
        ):
            return None
        return self._symbol_from_child_info(info.child_nodes[attr], mod)

    def _symbol_from_child_info(
        self, info: typeshed_client.resolver.ResolvedName, mod: str
    ) -> ClassSymbol | None:
        if info is None:
            return None
        if isinstance(info, typeshed_client.ImportedInfo):
            return self._symbol_from_child_info(info.info, ".".join(info.source_module))
        if not isinstance(info, typeshed_client.NameInfo):
            return None
        node = info.ast
        if isinstance(node, ast.AnnAssign):
            return self._symbol_from_annassign(node, mod)
        if isinstance(node, ast.Assign):
            return self._symbol_from_assign(node, mod)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._symbol_from_function_node(node, mod)
        if isinstance(node, typeshed_client.OverloadedName):
            return self._symbol_from_overloaded_node(node, mod)
        if isinstance(node, ast.ClassDef):
            return ClassSymbol(initializer=AnyValue(AnySource.inference))
        return None

    def _symbol_from_annassign(
        self, node: ast.AnnAssign, mod: str
    ) -> ClassSymbol | None:
        expr = self._parse_annotation(node.annotation, mod)
        annotation, qualifiers = expr.maybe_unqualify(_CLASS_SYMBOL_ALLOWED_QUALIFIERS)
        initializer = self._initializer_from_stub_assignment(node.value, mod)
        return ClassSymbol(
            annotation=(
                annotation
                if annotation is not None
                else AnyValue(AnySource.incomplete_annotation)
            ),
            qualifiers=frozenset(qualifiers),
            is_instance_only=(
                Qualifier.ClassVar not in qualifiers
                and Qualifier.InitVar not in qualifiers
            ),
            initializer=initializer,
        )

    def _symbol_from_assign(self, node: ast.Assign, mod: str) -> ClassSymbol | None:
        return ClassSymbol(
            initializer=(
                self._initializer_from_stub_assignment(node.value, mod)
                or AnyValue(AnySource.inference)
            )
        )

    def _symbol_from_function_node(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, mod: str
    ) -> ClassSymbol | None:
        is_property, is_classmethod, is_staticmethod, qualifiers, deprecated = (
            self._analyze_stub_method_decorators(node, mod)
        )
        if is_property:
            getter_type = (
                self._parse_type(node.returns, mod)
                if node.returns is not None
                else AnyValue(AnySource.unannotated)
            )
            return ClassSymbol(
                qualifiers=qualifiers,
                property_info=PropertyInfo(
                    getter_type=getter_type, getter_deprecation=deprecated
                ),
                initializer=TypedValue(property),
            )
        sig = self._get_signature_from_func_def(node, None, mod)
        initializer: Value
        if sig is None:
            initializer = AnyValue(AnySource.inference)
        else:
            initializer = CallableValue(sig)
            if deprecated is not None:
                initializer = annotate_value(
                    initializer, [DeprecatedExtension(deprecated)]
                )
        return ClassSymbol(
            qualifiers=qualifiers,
            is_method=True,
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
            initializer=initializer,
        )

    def _symbol_from_overloaded_node(
        self, node: typeshed_client.OverloadedName, mod: str
    ) -> ClassSymbol | None:
        method_nodes = [
            defn
            for defn in node.definitions
            if isinstance(defn, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if not method_nodes:
            return None
        is_property, is_classmethod, is_staticmethod, qualifiers, deprecated = (
            self._analyze_stub_method_decorators(method_nodes[0], mod)
        )
        if is_property:
            getter_type: Value = AnyValue(AnySource.inference)
            if all(defn.returns is not None for defn in method_nodes):
                getter_type = unite_values(
                    *(self._parse_type(defn.returns, mod) for defn in method_nodes)
                )
            return ClassSymbol(
                qualifiers=qualifiers,
                property_info=PropertyInfo(
                    getter_type=getter_type, getter_deprecation=deprecated
                ),
                initializer=TypedValue(property),
            )
        value = self._get_value_from_child_info(
            node, mod, on_class=False, parent_name="<overload>"
        )
        if isinstance(value, AnnotationExpr):
            value, _ = value.unqualify()
        if deprecated is not None:
            value = annotate_value(value, [DeprecatedExtension(deprecated)])
        return ClassSymbol(
            qualifiers=qualifiers,
            is_method=True,
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
            initializer=value,
        )

    def _analyze_stub_method_decorators(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, mod: str
    ) -> tuple[bool, bool, bool, frozenset[Qualifier], str | None]:
        is_property = False
        is_classmethod = False
        is_staticmethod = False
        qualifiers: set[Qualifier] = set()
        deprecated = None
        for decorator_node in node.decorator_list:
            decorator_value = self._parse_expr(decorator_node, mod)
            if decorator_value in PROPERTY_LIKE:
                is_property = True
            elif decorator_value == KnownValue(classmethod):
                is_classmethod = True
            elif decorator_value == KnownValue(staticmethod):
                is_staticmethod = True
            elif isinstance(decorator_value, KnownValue) and is_typing_name(
                decorator_value.val, "final"
            ):
                qualifiers.add(Qualifier.Final)
            elif isinstance(
                extension := self._extract_extension_from_decorator(decorator_value),
                DeprecatedExtension,
            ):
                deprecated = extension.deprecation_message
        return (
            is_property,
            is_classmethod,
            is_staticmethod,
            frozenset(qualifiers),
            deprecated,
        )

    def _initializer_from_stub_assignment(
        self, node: ast.AST | None, mod: str
    ) -> Value | None:
        if node is None:
            return None
        value = self._parse_expr(node, mod)
        if value == KnownValue(...):
            return None
        return value

    def _has_own_attribute(self, typ: type | str, attr: str) -> bool:
        # Special case since otherwise we think every object has every attribute
        if typ is object and attr == "__getattribute__":
            return False
        if isinstance(typ, str):
            fq_name = typ
        else:
            fq_name = self._get_fq_name(typ)
            if fq_name is None:
                return False
        info = self._get_info_for_name(fq_name)
        mod, _ = fq_name.rsplit(".", maxsplit=1)
        return self._has_attribute_from_info(info, mod, attr)

    def _get_all_attributes_from_info(
        self, info: typeshed_client.resolver.ResolvedName, mod: str
    ) -> set[str]:
        if info is None:
            return set()
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_all_attributes_from_info(
                info.info, ".".join(info.source_module)
            )
        elif isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, ast.ClassDef):
                if info.child_nodes is not None:
                    return set(info.child_nodes)
            elif isinstance(info.ast, ast.Assign):
                val = self._parse_expr(info.ast.value, mod)
                if isinstance(val, KnownValue) and isinstance(val.val, type):
                    return self.get_all_attributes(val.val)
                else:
                    return set()
            else:
                return set()
        return set()

    def _has_attribute_from_info(
        self, info: typeshed_client.resolver.ResolvedName, mod: str, attr: str
    ) -> bool:
        if info is None:
            return False
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._has_attribute_from_info(
                info.info, ".".join(info.source_module), attr
            )
        elif isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, ast.ClassDef):
                if info.child_nodes and attr in info.child_nodes:
                    return True
                return False
            elif isinstance(info.ast, ast.Assign):
                val = self._parse_expr(info.ast.value, mod)
                if isinstance(val, KnownValue) and isinstance(val.val, type):
                    return self.has_attribute(val.val, attr)
                else:
                    return False
            else:
                return False
        return False

    def _get_bases_from_info(
        self, info: typeshed_client.resolver.ResolvedName, mod: str, fq_name: str
    ) -> list[Value] | None:
        if info is None:
            return None
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_bases_from_info(
                info.info, ".".join(info.source_module), fq_name
            )
        elif isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, ast.ClassDef):
                bases = info.ast.bases
                return [self._parse_type(base, mod) for base in bases]
            elif isinstance(info.ast, ast.AnnAssign):
                if info.ast.value is not None:
                    val = self._parse_expr(info.ast.value, mod)
                    if isinstance(val, KnownValue) and isinstance(val.val, type):
                        new_fq_name = self._get_fq_name(val.val)
                        if fq_name == new_fq_name:
                            # prevent infinite recursion
                            return [AnyValue(AnySource.inference)]
                        return self.get_bases(val.val)
                # Stubs model some typing special forms this way (for example
                # `Annotated: _SpecialForm`), which do not have meaningful bases.
                return [AnyValue(AnySource.inference)]
            elif isinstance(info.ast, ast.Assign):
                val = self._parse_expr(info.ast.value, mod)
                if isinstance(val, KnownValue) and isinstance(val.val, type):
                    new_fq_name = self._get_fq_name(val.val)
                    if fq_name == new_fq_name:
                        # prevent infinite recursion
                        return [AnyValue(AnySource.inference)]
                    return self.get_bases(val.val)
                else:
                    return [AnyValue(AnySource.inference)]
            elif isinstance(
                info.ast,
                (
                    # overloads are not supported yet
                    typeshed_client.OverloadedName,
                    typeshed_client.ImportedName,
                    # typeshed pretends the class is a function
                    ast.FunctionDef,
                ),
            ):
                return None
            else:
                raise NotImplementedError(ast.dump(info.ast))
        return None

    def _get_method_signature_from_info(
        self,
        info: typeshed_client.resolver.ResolvedName,
        obj: object,
        fq_name: str,
        mod: str,
        objclass: type,
        *,
        allow_call: bool = False,
    ) -> ConcreteSignature | None:
        if info is None:
            return None
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_method_signature_from_info(
                info.info,
                obj,
                fq_name,
                ".".join(info.source_module),
                objclass,
                allow_call=allow_call,
            )
        elif isinstance(info, typeshed_client.NameInfo):
            # Note that this doesn't handle names inherited from base classes
            if info.child_nodes and obj.__name__ in info.child_nodes:
                child_info = info.child_nodes[obj.__name__]
                return self._get_signature_from_info(
                    child_info, obj, fq_name, mod, objclass, allow_call=allow_call
                )
            else:
                return None
        else:
            self.log("Ignoring unrecognized info", (fq_name, info))
            return None

    def _get_fq_name(self, obj: Any) -> str | None:
        if obj is GeneratorType:
            return "typing.Generator"
        # It claims to be io.open, but typeshed puts it in builtins
        if obj is open:
            return "builtins.open"
        try:
            module_name = obj.__module__
            if module_name is None:
                module_name = "builtins"
            fq_name = ".".join([module_name, obj.__qualname__])
            # Avoid looking for stubs we won't find anyway.
            if not _obj_from_qualname_is(module_name, obj.__qualname__, obj):
                self.log("Ignoring invalid name", fq_name)
                return None
            return _TYPING_ALIASES.get(fq_name, fq_name)
        except (AttributeError, TypeError):
            self.log("Ignoring object without module or qualname", obj)
            return None

    def _sig_from_value(self, val: Value) -> ConcreteSignature | None:
        if isinstance(val, UninitializedValue):
            return None
        val, extensions = unannotate_value(val, DeprecatedExtension)
        val = replace_fallback(val)
        if not isinstance(val, CallableValue):
            return None
        sig = val.signature
        if isinstance(sig, Signature):
            for extension in extensions:
                sig = replace(sig, deprecated=extension.deprecation_message)
        return sig

    def _get_signature_from_info(
        self,
        info: typeshed_client.resolver.ResolvedName,
        obj: object,
        fq_name: str,
        mod: str,
        objclass: type | None = None,
        *,
        allow_call: bool = False,
        type_params: Sequence[TypeParam] = (),
    ) -> ConcreteSignature | None:
        if isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return self._get_signature_from_func_def(
                    info.ast, obj, mod, objclass, allow_call=allow_call
                )
            elif isinstance(info.ast, typeshed_client.OverloadedName):
                sigs = []
                for defn in info.ast.definitions:
                    if not isinstance(defn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.log(
                            "Ignoring unrecognized AST in overload", (fq_name, info)
                        )
                        return None
                    sig = self._get_signature_from_func_def(
                        defn, obj, mod, objclass, allow_call=allow_call
                    )
                    if sig is None:
                        self.log("Could not get sig for overload member", (defn,))
                        return None
                    sigs.append(sig)
                return OverloadedSignature(sigs)
            elif isinstance(info.ast, ast.ClassDef):
                new_value, provider = self.get_attribute_recursively(
                    fq_name, "__new__", on_class=True
                )
                from_init = False
                if new_value is UNINITIALIZED_VALUE or provider is object:
                    init_value, provider = self.get_attribute_recursively(
                        fq_name, "__init__", on_class=True
                    )
                    if (sig := self._sig_from_value(init_value)) is not None:
                        from_init = True
                else:
                    sig = self._sig_from_value(new_value)
                if sig is not None:
                    if safe_isinstance(obj, type):
                        if allow_call:
                            if isinstance(sig, Signature):
                                sig = replace(sig, allow_call=True, callable=obj)
                            else:
                                sig = OverloadedSignature(
                                    [
                                        replace(sig, allow_call=True, callable=obj)
                                        for sig in sig.signatures
                                    ]
                                )
                        typ = obj
                    else:
                        typ = fq_name
                    if type_params:
                        self_val = GenericValue(
                            typ,
                            [
                                type_param_to_value(type_param)
                                for type_param in type_params
                            ],
                        )
                    else:
                        self_val = TypedValue(typ)
                    if from_init:
                        sig = sig.replace_return_value(self_val)
                        self_annotation_value = self_val
                    else:
                        self_annotation_value = SubclassValue(self_val)
                    bound_sig = make_bound_method(
                        sig, Composite(self_val), ctx=self.ctx
                    )
                    if bound_sig is None:
                        return None
                    sig = bound_sig.get_signature(
                        ctx=self.ctx, self_annotation_value=self_annotation_value
                    )
                    return sig

                return None
            else:
                self.log("Ignoring unrecognized AST", (fq_name, info))
                return None
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_signature_from_info(
                info.info,
                obj,
                fq_name,
                ".".join(info.source_module),
                objclass,
                allow_call=allow_call,
            )
        elif info is None:
            return None
        else:
            self.log("Ignoring unrecognized info", (fq_name, info))
            return None

    def _get_info_for_name(self, fq_name: str) -> typeshed_client.resolver.ResolvedName:
        if fq_name not in self._info_cache:
            self._info_cache[fq_name] = self.resolver.get_fully_qualified_name(fq_name)
        return self._info_cache[fq_name]

    def _get_signature_from_func_def(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        obj: object,
        mod: str,
        objclass: type | None = None,
        *,
        autobind: bool = False,
        bind_classmethod: bool = False,
        allow_call: bool = False,
    ) -> Signature | None:
        is_classmethod = is_staticmethod = is_evaluated = False
        deprecated = None
        for decorator_ast in node.decorator_list:
            decorator = self._parse_expr(decorator_ast, mod)
            if (
                decorator == KnownValue(abstractmethod)
                or decorator == KnownValue(overload)
                or decorator == KnownValue(real_overload)
            ):
                continue
            elif decorator == KnownValue(classmethod):
                is_classmethod = True
                if autobind:  # TODO support classmethods otherwise
                    continue
            elif decorator == KnownValue(staticmethod):
                is_staticmethod = True
                if autobind:  # TODO support staticmethods otherwise
                    continue
            elif decorator == KnownValue(evaluated):
                is_evaluated = True
                continue
            elif (
                isinstance(decorator, DecoratorValue)
                and decorator.decorator is deprecated_decorator
            ):
                arg = decorator.args[0]
                if isinstance(arg, KnownValue) and isinstance(arg.val, str):
                    deprecated = arg.val
            # something we don't recognize; ignore it
        if node.returns is None:
            return_value = AnyValue(AnySource.unannotated)
        else:
            return_value = self._parse_type(node.returns, mod)
        # ignore self type for class and static methods
        if is_classmethod or is_staticmethod:
            objclass = None
        args = node.args
        arguments: list[SigParameter] = []
        num_pos_only_args = len(args.posonlyargs)
        defaults = args.defaults
        num_pos_only_defaults = len(defaults) - len(args.args)
        if num_pos_only_defaults > 0:
            num_without_default = num_pos_only_args - num_pos_only_defaults
            pos_only_defaults = [None] * num_without_default + defaults[
                :num_pos_only_defaults
            ]
            defaults = defaults[num_pos_only_defaults:]
        else:
            pos_only_defaults = [None for _ in args.posonlyargs]
        arguments += self._parse_param_list(
            args.posonlyargs,
            pos_only_defaults,
            mod,
            ParameterKind.POSITIONAL_ONLY,
            objclass,
        )

        num_without_defaults = len(args.args) - len(defaults)
        defaults = [None] * num_without_defaults + defaults
        arguments += self._parse_param_list(
            args.args, defaults, mod, ParameterKind.POSITIONAL_OR_KEYWORD, objclass
        )
        if autobind:
            if is_classmethod or not is_staticmethod:
                arguments = arguments[1:]
        elif bind_classmethod and is_classmethod:
            # Access via class should bind classmethods, but not regular methods.
            arguments = arguments[1:]

        if args.vararg is not None:
            arguments.append(
                self._parse_param(args.vararg, None, mod, ParameterKind.VAR_POSITIONAL)
            )
        arguments += self._parse_param_list(
            args.kwonlyargs, args.kw_defaults, mod, ParameterKind.KEYWORD_ONLY
        )
        if args.kwarg is not None:
            arguments.append(
                self._parse_param(args.kwarg, None, mod, ParameterKind.VAR_KEYWORD)
            )
        # some typeshed types have a positional-only after a normal argument,
        # and Signature doesn't like that
        seen_non_positional = False
        cleaned_arguments = []
        for arg in arguments:
            if arg.kind is ParameterKind.POSITIONAL_ONLY and seen_non_positional:
                cleaned_arguments = [
                    replace(arg, kind=ParameterKind.POSITIONAL_ONLY)
                    for arg in cleaned_arguments
                ]
                seen_non_positional = False
            else:
                seen_non_positional = True
            cleaned_arguments.append(arg)
        if is_evaluated:
            ctx = _AnnotationContext(self, mod)
            evaluator = SyntheticEvaluator(node, return_value, ctx)
        else:
            evaluator = None
        return Signature.make(
            cleaned_arguments,
            callable=obj,
            return_annotation=(
                make_coro_type(return_value)
                if isinstance(node, ast.AsyncFunctionDef)
                else return_value
            ),
            allow_call=allow_call,
            evaluator=evaluator,
            deprecated=deprecated,
        )

    def _parse_param_list(
        self,
        args: Iterable[ast.arg],
        defaults: Iterable[ast.AST | None],
        module: str,
        kind: ParameterKind,
        objclass: type | None = None,
    ) -> Iterable[SigParameter]:
        for i, (arg, default) in enumerate(zip(args, defaults)):
            yield self._parse_param(
                arg, default, module, kind, objclass=objclass if i == 0 else None
            )

    def _parse_param(
        self,
        arg: ast.arg,
        default: ast.AST | None,
        module: str,
        kind: ParameterKind,
        *,
        objclass: type | None = None,
    ) -> SigParameter:
        typ: Value | AnnotationExpr = AnyValue(AnySource.unannotated)
        if arg.annotation is not None:
            typ = self._parse_annotation(arg.annotation, module)
        elif objclass is not None:
            bases = self.get_bases(objclass)
            if bases is None:
                typ = TypedValue(objclass)
            else:
                typevars = uniq_chain(
                    tuple(iter_type_params_in_value(base)) for base in bases
                )
                if typevars:
                    typ = GenericValue(
                        objclass,
                        [
                            (
                                TypeVarValue(type_param)
                                if isinstance(type_param, TypeVarParam)
                                else (
                                    TypeVarTupleValue(type_param)
                                    if isinstance(type_param, TypeVarTupleParam)
                                    else InputSigValue(type_param)
                                )
                            )
                            for type_param in typevars
                        ],
                    )
                else:
                    typ = TypedValue(objclass)

        name = arg.arg
        if kind is ParameterKind.POSITIONAL_OR_KEYWORD and is_positional_only_arg_name(
            name
        ):
            kind = ParameterKind.POSITIONAL_ONLY
            name = name[2:]
        typ = translate_vararg_type(kind, typ, self.ctx)
        # Mark self as positional-only. objclass should be given only if we believe
        # it's the "self" parameter.
        if objclass is not None:
            kind = ParameterKind.POSITIONAL_ONLY
        if default is None:
            return SigParameter(name, kind, annotation=typ)
        else:
            default_value = self._parse_expr(default, module)
            if default_value == KnownValue(...):
                default_value = AnyValue(AnySource.unannotated)
            return SigParameter(name, kind, annotation=typ, default=default_value)

    def _parse_expr(self, node: ast.AST, module: str) -> Value:
        ctx = _AnnotationContext(finder=self, module=module)
        return value_from_ast(node, ctx=ctx)

    def _parse_annotation(self, node: ast.AST, module: str) -> AnnotationExpr:
        val = self._parse_expr(node, module)
        ctx = _AnnotationContext(finder=self, module=module)
        expr = annotation_expr_from_value(val, ctx=ctx)
        return expr

    def _parse_type(self, node: ast.AST, module: str) -> Value:
        expr = self._parse_annotation(node, module)
        val, _ = expr.unqualify()
        if self.verbose and isinstance(val, AnyValue):
            self.log("Got Any", (ast.dump(node), module))
        return val

    def _parse_call_assignment(
        self, info: typeshed_client.NameInfo, module: str
    ) -> Value:
        try:
            __import__(module)
            mod = sys.modules[module]
            return KnownValue(getattr(mod, info.name))
        except Exception:
            pass

        if not isinstance(info.ast, ast.Assign) or not isinstance(
            info.ast.value, ast.Call
        ):
            return AnyValue(AnySource.inference)
        ctx = _AnnotationContext(finder=self, module=module)
        return value_from_ast(info.ast.value, ctx=ctx)

    def _extract_metadata(self, module: str, node: ast.ClassDef) -> Sequence[Extension]:
        metadata = []
        for decorator in node.decorator_list:
            decorator_val = self._parse_expr(decorator, module)
            extension = self._extract_extension_from_decorator(decorator_val)
            if extension is not None:
                metadata.append(extension)
        return metadata

    def _extract_extension_from_decorator(
        self, decorator_val: Value
    ) -> Extension | None:
        if (
            isinstance(decorator_val, DecoratorValue)
            and decorator_val.decorator is deprecated_decorator
        ):
            arg = decorator_val.args[0]
            if isinstance(arg, KnownValue) and isinstance(arg.val, str):
                return DeprecatedExtension(arg.val)
        return None

    def make_synthetic_type(self, module: str, info: typeshed_client.NameInfo) -> Value:
        fq_name = f"{module}.{info.name}"
        bases = self._get_bases_from_info(info, module, fq_name)
        typ = TypedValue(fq_name)
        if isinstance(info.ast, ast.ClassDef):
            metadata = self._extract_metadata(module, info.ast)
        else:
            metadata = []
        if bases is not None:
            if any(
                (isinstance(base, KnownValue) and is_typing_name(base.val, "TypedDict"))
                or isinstance(base, TypedDictValue)
                for base in bases
            ):
                typ = self._make_typeddict(module, info, bases)
        val = SyntheticClassObjectValue(info.name, typ)
        if metadata:
            return annotate_value(val, metadata)
        return val

    def _make_typeddict(
        self, module: str, info: typeshed_client.NameInfo, bases: Sequence[Value]
    ) -> TypedDictValue:
        total = True
        if isinstance(info.ast, ast.ClassDef):
            for keyword in info.ast.keywords:
                # TODO support PEP 728 here
                if keyword.arg == "total":
                    val = self._parse_expr(keyword.value, module)
                    if isinstance(val, KnownValue) and isinstance(val.val, bool):
                        total = val.val
        attrs = self._get_all_attributes_from_info(info, module)
        fields = [
            self._get_attribute_from_info(info, module, attr, on_class=True)
            for attr in attrs
        ]
        items = {}
        for base in bases:
            if isinstance(base, TypedDictValue):
                items.update(base.items)
        items.update(
            {
                attr: self._make_td_value(field, total)
                for attr, field in zip(attrs, fields)
            }
        )
        return TypedDictValue(items)

    def _make_td_value(
        self, field: Value | AnnotationExpr, total: bool
    ) -> TypedDictEntry:
        readonly = False
        required = total
        if isinstance(field, AnnotationExpr):
            field, qualifiers = field.unqualify(
                {Qualifier.Required, Qualifier.ReadOnly, Qualifier.NotRequired},
                mutually_exclusive_qualifiers=(
                    (Qualifier.Required, Qualifier.NotRequired),
                ),
            )
            if Qualifier.ReadOnly in qualifiers:
                readonly = True
            if Qualifier.Required in qualifiers:
                required = True
            if Qualifier.NotRequired in qualifiers:
                required = False
        return TypedDictEntry(readonly=readonly, required=required, typ=field)

    def _value_from_info(
        self, info: typeshed_client.resolver.ResolvedName, module: str
    ) -> Value:
        # This guard against infinite recursion if a type refers to itself
        # (real-world example: os._ScandirIterator). Needs to change in
        # order to support recursive types.
        if info in self._active_infos:
            return AnyValue(AnySource.inference)
        self._active_infos.append(info)
        try:
            return self._value_from_info_inner(info, module)
        finally:
            self._active_infos.pop()

    def _value_from_info_inner(
        self, info: typeshed_client.resolver.ResolvedName, module: str
    ) -> Value:
        if isinstance(info, typeshed_client.ImportedInfo):
            return self._value_from_info(info.info, ".".join(info.source_module))
        elif isinstance(info, typeshed_client.NameInfo):
            fq_name = f"{module}.{info.name}"
            if fq_name in _TYPING_ALIASES:
                new_fq_name = _TYPING_ALIASES[fq_name]
                info = self._get_info_for_name(new_fq_name)
                return self._value_from_info(
                    info, new_fq_name.rsplit(".", maxsplit=1)[0]
                )
            if isinstance(info.ast, ast.Assign):
                key = (module, info.ast)
                if key in self._assignment_cache:
                    return self._assignment_cache[key]
                if isinstance(info.ast.value, ast.Call):
                    value = self._parse_call_assignment(info, module)
                else:
                    value = self._parse_expr(info.ast.value, module)
                self._assignment_cache[key] = value
                return value
            try:
                __import__(module)
                mod = sys.modules[module]
                return KnownValue(getattr(mod, info.name))
            except Exception:
                if isinstance(info.ast, ast.ClassDef):
                    return self.make_synthetic_type(module, info)
                elif isinstance(info.ast, ast.AnnAssign):
                    expr = self._parse_annotation(info.ast.annotation, module)
                    val, qualifiers = expr.maybe_unqualify({Qualifier.TypeAlias})
                    if val is not None and Qualifier.TypeAlias not in qualifiers:
                        return val
                    if info.ast.value:
                        return self._parse_expr(info.ast.value, module)
                elif isinstance(
                    info.ast,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        typeshed_client.OverloadedName,
                    ),
                ):
                    sig = self._get_signature_from_info(info, None, fq_name, module)
                    if sig is not None:
                        return CallableValue(sig)
                self.log("Unable to import", (module, info))
                return AnyValue(AnySource.inference)
        elif isinstance(info, tuple):
            module_path = ".".join(info)
            try:
                __import__(module_path)
                return KnownValue(sys.modules[module_path])
            except Exception:
                return SyntheticModuleValue(info)
        else:
            self.log("Ignoring info", info)
            return AnyValue(AnySource.inference)


def _obj_from_qualname_is(module_name: str, qualname: str, obj: object) -> bool:
    try:
        if module_name not in sys.modules:
            __import__(module_name)
        mod = sys.modules[module_name]
        actual = mod
        for piece in qualname.split("."):
            actual = getattr(actual, piece)
        return obj is actual
    except Exception:
        return False
