# static analysis: ignore

from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestDataclassTransform(TestNameCheckVisitorBase):
    @assert_passes()
    def test_non_typing_decorator_named_dataclass_transform(self):
        from typing import Callable, TypeVar

        T = TypeVar("T")

        def dataclass_transform() -> Callable[[T], T]:
            def decorator(cls: T) -> T:
                return cls

            return decorator

        @dataclass_transform()
        def model(cls: type[T]) -> type[T]:
            return cls

        @model
        class Customer:
            id: int
            name: str

        def check_errors() -> None:
            Customer(id=1, name="")  # E: incompatible_call

    @assert_passes()
    def test_non_typing_class_decorator_named_dataclass_transform(self):
        def dataclass_transform():
            def decorator(cls: type[object]) -> type[object]:
                return cls

            return decorator

        @dataclass_transform()
        class BaseModel:
            pass

        class Customer(BaseModel):
            id: int

        def check_errors() -> None:
            Customer(id=1)  # E: incompatible_call

    @assert_passes()
    def test_dataclass_transform_function_provider_in_same_module(self):
        from dataclasses import dataclass
        from typing import TypeVar

        from typing_extensions import dataclass_transform

        T = TypeVar("T")

        @dataclass_transform()
        def model(cls: type[T]) -> type[T]:
            return dataclass(cls)

        @model
        class Customer:
            id: int
            name: str

        Customer(id=1, name="")

        def check_errors() -> None:
            Customer(name="")  # E: incompatible_call
            Customer(id=1, unknown="")  # E: incompatible_call

    @assert_passes()
    def test_dataclass_transform_overload_provider_in_same_module(self):
        from dataclasses import dataclass
        from typing import Any, Callable, TypeVar, overload

        from typing_extensions import dataclass_transform

        T = TypeVar("T")

        @overload
        @dataclass_transform(kw_only_default=True)
        def model(cls: T) -> T: ...

        @overload
        def model(*, kw_only: bool = True) -> Callable[[T], T]: ...

        def model(*args: Any, **kwargs: Any) -> Any:
            def decorator(cls: type[T]) -> type[T]:
                return dataclass(cls, kw_only=kwargs.get("kw_only", True))

            if args and len(args) == 1 and isinstance(args[0], type):
                return decorator(args[0])
            return decorator

        @model
        class Customer1:
            id: int
            name: str

        @model(kw_only=False)
        class Customer2:
            id: int
            name: str

        Customer1(id=1, name="")
        Customer2(1, "")

        def check_errors() -> None:
            Customer1(1, "")  # E: incompatible_call

    @assert_passes()
    def test_dataclass_transform_decorator_base_and_metaclass(self):
        from dataclasses import dataclass
        from typing import Callable, TypeVar

        from typing_extensions import dataclass_transform

        T = TypeVar("T")

        @dataclass_transform(kw_only_default=True, frozen_default=True)
        def create_model(
            *, frozen: bool = True, kw_only: bool = True
        ) -> Callable[[type[T]], type[T]]:
            def decorator(cls: type[T]) -> type[T]:
                return dataclass(cls, frozen=frozen, kw_only=kw_only)

            return decorator

        @create_model()
        class Decorated:
            x: int

        decorated = Decorated(x=1)

        @dataclass_transform(kw_only_default=True, frozen_default=True)
        class BaseModel:
            def __init_subclass__(
                cls, *, frozen: bool = True, kw_only: bool = True
            ) -> None:
                dataclass(cls, frozen=frozen, kw_only=kw_only)

        class FromBase(BaseModel, frozen=True):
            a: int

        FromBase(a=1)

        @dataclass_transform(kw_only_default=True, frozen_default=True)
        class ModelMeta(type):
            def __new__(
                mcls,
                name: str,
                bases: tuple[type, ...],
                namespace: dict[str, object],
                *,
                frozen: bool = True,
                kw_only: bool = True,
            ) -> type:
                cls = super().__new__(mcls, name, bases, namespace)
                return dataclass(cls, frozen=frozen, kw_only=kw_only)

        class WithMeta(metaclass=ModelMeta, frozen=True): ...

        class FromMeta(WithMeta, frozen=True):
            b: int

        FromMeta(b=1)

        def check_errors() -> None:
            Decorated(1)  # E: incompatible_call
            decorated.x = 3  # E: incompatible_assignment
            FromBase(1)  # E: incompatible_call
            from_base = FromBase(a=1)
            from_base.a = 2  # E: incompatible_assignment
            FromMeta(1)  # E: incompatible_call
            from_meta = FromMeta(b=1)
            from_meta.b = 2  # E: incompatible_assignment

    @assert_passes(allow_import_failures=True)
    def test_dataclass_transform_synthetic_classes_after_import_failure(self):
        from typing import Callable, TypeVar

        from typing_extensions import dataclass_transform

        T = TypeVar("T")

        def model_field(*, init: bool = True, default: object = 0) -> object:
            raise NotImplementedError

        @dataclass_transform(
            kw_only_default=True, frozen_default=True, field_specifiers=(model_field,)
        )
        def create_model(
            *, frozen: bool = True, kw_only: bool = True
        ) -> Callable[[type[T]], type[T]]:
            def decorator(cls: type[T]) -> type[T]:
                return cls

            return decorator

        @create_model()
        class Decorated:
            x: int = model_field(init=False)
            y: int = model_field(default=3)

        Decorated(y=1)
        Decorated(x=1, y=1)  # E: incompatible_call
        Decorated(1)  # E: incompatible_call
        decorated = Decorated(y=1)
        decorated.y = 2  # E: incompatible_assignment

        @create_model(frozen=False)
        class BadDecoratedChild(Decorated):  # E: invalid_base
            z: int

        @dataclass_transform(kw_only_default=True, frozen_default=True)
        class BaseModel:
            def __init_subclass__(
                cls, *, frozen: bool = True, kw_only: bool = True
            ) -> None:
                pass

        class BaseDerived(BaseModel, frozen=True):
            a: int

        BaseDerived(a=1)
        BaseDerived(1)  # E: incompatible_call
        base_derived = BaseDerived(a=1)
        base_derived.a = 2  # E: incompatible_assignment

        @dataclass_transform(kw_only_default=True, frozen_default=True)
        class ModelMeta(type):
            pass

        class WithMeta(metaclass=ModelMeta):
            def __init_subclass__(
                cls, *, frozen: bool = True, kw_only: bool = True
            ) -> None:
                pass

        class MetaDerived(WithMeta, frozen=True):
            b: int

        MetaDerived(b=1)
        MetaDerived(1)  # E: incompatible_call
        meta_derived = MetaDerived(b=1)
        meta_derived.b = 2  # E: incompatible_assignment

    @assert_passes(allow_import_failures=True)
    def test_dataclass_transform_slots_semantics_after_import_failure(self):
        from typing import Callable, TypeVar

        from typing_extensions import dataclass_transform

        T = TypeVar("T")

        @dataclass_transform()
        def model(*, slots: bool = False) -> Callable[[type[T]], type[T]]:
            def decorator(cls: type[T]) -> type[T]:
                return cls

            return decorator

        @model(slots=True)
        class Slotted:
            x: int

        Slotted.__slots__
        Slotted(1).__slots__
        slotted = Slotted(1)
        slotted.y = 3  # E: incompatible_assignment

        @model()
        class NotSlotted:
            x: int

        NotSlotted.__slots__  # E: undefined_attribute
        NotSlotted(1).__slots__  # E: undefined_attribute

        @model(slots=True)
        class BadModel:  # E: invalid_dataclass
            x: int
            __slots__ = ()

        @dataclass_transform()
        class BaseModel:
            pass

        class FromBase(BaseModel, slots=True):
            y: int

        FromBase.__slots__
        FromBase(1).__slots__

    @assert_passes()
    def test_dataclass_transform_fully_imported_providers(self):
        import sys
        import types
        from typing import Any, Callable, TypeVar

        from typing_extensions import dataclass_transform

        T = TypeVar("T")
        helper = types.ModuleType("dt_transform_helpers")

        def model_field(
            *,
            init: bool = True,
            default: Any = 0,
            alias: str | None = None,
            kw_only: bool = False,
        ) -> Any:
            return object()

        @dataclass_transform(
            kw_only_default=True, frozen_default=True, field_specifiers=(model_field,)
        )
        def create_model(
            *, frozen: bool = True, kw_only: bool = True
        ) -> Callable[[type[T]], type[T]]:
            def decorator(cls: type[T]) -> type[T]:
                return cls

            return decorator

        @dataclass_transform(
            kw_only_default=True, frozen_default=False, field_specifiers=(model_field,)
        )
        class BaseModel:
            def __init_subclass__(
                cls, *, frozen: bool = False, kw_only: bool = True
            ) -> None:
                pass

        @dataclass_transform(
            kw_only_default=True, frozen_default=False, field_specifiers=(model_field,)
        )
        class ModelMeta(type):
            pass

        helper.model_field = model_field
        helper.create_model = create_model
        helper.BaseModel = BaseModel
        helper.ModelMeta = ModelMeta
        sys.modules["dt_transform_helpers"] = helper

        from dt_transform_helpers import BaseModel as ImportedBaseModel
        from dt_transform_helpers import ModelMeta as ImportedModelMeta
        from dt_transform_helpers import create_model as imported_create_model
        from dt_transform_helpers import model_field as imported_model_field

        @imported_create_model()
        class Decorated:
            x: int = imported_model_field(init=False)
            y: int = imported_model_field(alias="why")
            z: int = imported_model_field(default=3, kw_only=False)

        class FromBase(ImportedBaseModel, frozen=True):
            a: int = imported_model_field(init=False)
            b: int = imported_model_field(alias="bee")
            c: int = imported_model_field(default=3, kw_only=False)

        class WithMeta(metaclass=ImportedModelMeta):
            def __init_subclass__(
                cls, *, frozen: bool = False, kw_only: bool = True
            ) -> None:
                pass

        class FromMeta(WithMeta, frozen=True):
            m: int = imported_model_field(init=False)
            n: int = imported_model_field(alias="enn")
            o: int = imported_model_field(default=3, kw_only=False)

        class MutableFromMeta(FromMeta, frozen=False):  # E: invalid_base
            p: int = imported_model_field(alias="pee")

        def check_calls() -> None:
            Decorated(2, why=1)
            Decorated(2, y=1)  # E: incompatible_call
            Decorated(x=1, why=1)  # E: incompatible_call
            decorated = Decorated(z=3, why=1)
            decorated.y = 4  # E: incompatible_assignment

            FromBase(2, bee=1)
            FromBase(2, b=1)  # E: incompatible_call
            from_base = FromBase(c=2, bee=1)
            from_base.b = 4  # E: incompatible_assignment

            FromMeta(2, enn=1)
            FromMeta(2, n=1)  # E: incompatible_call
            from_meta = FromMeta(o=2, enn=1)
            from_meta.n = 4  # E: incompatible_assignment

    @assert_passes()
    def test_dataclass_transform_decorator_field_specifier_options(self):
        from typing import Any, Callable, TypeVar

        from typing_extensions import dataclass_transform

        T = TypeVar("T")

        def model_field(
            *,
            init: bool = True,
            default: Any = 0,
            alias: str | None = None,
            kw_only: bool = False,
        ) -> Any:
            return object()

        @dataclass_transform(
            kw_only_default=True, frozen_default=True, field_specifiers=(model_field,)
        )
        def create_model(
            *, frozen: bool = True, kw_only: bool = True
        ) -> Callable[[type[T]], type[T]]:
            def decorator(cls: type[T]) -> type[T]:
                return cls

            return decorator

        @create_model()
        class DecoratorModel:
            x: int = model_field(init=False)
            y: int = model_field(alias="why")
            z: int = model_field(default=3, kw_only=False)

        def check_calls() -> None:
            DecoratorModel(2, why=1)
            DecoratorModel(why=1)
            DecoratorModel(2, y=1)  # E: incompatible_call
            DecoratorModel(x=1, why=1)  # E: incompatible_call
            DecoratorModel(1, 2)  # E: incompatible_call
            model = DecoratorModel(2, why=1)
            model.y = 3  # E: incompatible_assignment

        def check_inheritance() -> None:
            @create_model(frozen=True)
            class Parent:
                p: int

            @create_model(frozen=False)
            class Child(Parent):  # E: invalid_base
                c: int

    @assert_passes()
    def test_dataclass_transform_field_specifier_implicit_defaults(self):
        from typing import Any, Callable, Literal, TypeVar, overload

        from typing_extensions import dataclass_transform

        T = TypeVar("T")

        @overload
        def field1(
            *,
            default: str | None = None,
            resolver: Callable[[], Any],
            init: Literal[False] = False,
        ) -> Any: ...

        @overload
        def field1(
            *,
            default: str | None = None,
            resolver: None = None,
            init: Literal[True] = True,
        ) -> Any: ...

        def field1(
            *,
            default: str | None = None,
            resolver: Callable[[], Any] | None = None,
            init: bool = True,
        ) -> Any:
            return object()

        def field2(*, init: bool = False, kw_only: bool = True) -> Any:
            return object()

        @dataclass_transform(kw_only_default=True, field_specifiers=(field1, field2))
        def create_model(*, init: bool = True) -> Callable[[type[T]], type[T]]:
            def decorator(cls: type[T]) -> type[T]:
                return cls

            return decorator

        @create_model()
        class CustomerModel1:
            id: int = field1(resolver=lambda: 0)
            name: str = field1(default="Voldemort")

        @create_model()
        class CustomerModel2:
            id: int = field2()
            name: str = field2(init=True)

        def check_calls() -> None:
            CustomerModel1()
            CustomerModel1(name="hi")
            CustomerModel1(id=1, name="hi")  # E: incompatible_call
            CustomerModel2(1)  # E: incompatible_call
            CustomerModel2(name="Fred")
            CustomerModel2(id=1, name="Fred")  # E: incompatible_call

    @assert_passes()
    def test_dataclass_transform_base_field_specifier_options(self):
        from typing import Any

        from typing_extensions import dataclass_transform

        def model_field(
            *,
            init: bool = True,
            default: Any = 0,
            alias: str | None = None,
            kw_only: bool = False,
        ) -> Any:
            return object()

        @dataclass_transform(
            kw_only_default=True, frozen_default=False, field_specifiers=(model_field,)
        )
        class BaseModel:
            def __init_subclass__(
                cls, *, frozen: bool = False, kw_only: bool = True
            ) -> None:
                pass

        class Concrete(BaseModel, frozen=True):
            a: int = model_field(init=False)
            b: int = model_field(alias="bee")
            c: int = model_field(default=3, kw_only=False)

        def check_calls() -> None:
            Concrete(2, bee=1)
            Concrete(bee=1)
            Concrete(2, b=1)  # E: incompatible_call
            Concrete(a=1, bee=1)  # E: incompatible_call
            Concrete(1, 2)  # E: incompatible_call
            model = Concrete(2, bee=1)
            model.b = 3  # E: incompatible_assignment

        def check_inheritance() -> None:
            class FrozenParent(BaseModel, frozen=True):
                x: int

            class MutableChild(FrozenParent, frozen=False):  # E: invalid_base
                y: int

    @assert_passes()
    def test_dataclass_transform_metaclass_field_specifier_options(self):
        from typing import Any

        from typing_extensions import dataclass_transform

        def model_field(
            *,
            init: bool = True,
            default: Any = 0,
            alias: str | None = None,
            kw_only: bool = False,
        ) -> Any:
            return object()

        @dataclass_transform(
            kw_only_default=True, frozen_default=False, field_specifiers=(model_field,)
        )
        class ModelMeta(type):
            pass

        class MetaBase(metaclass=ModelMeta):
            def __init_subclass__(
                cls, *, frozen: bool = False, kw_only: bool = True
            ) -> None:
                pass

        class MetaConcrete(MetaBase, frozen=True):
            a: int = model_field(init=False)
            b: int = model_field(alias="bee")
            c: int = model_field(default=3, kw_only=False)

        def check_calls() -> None:
            MetaConcrete(2, bee=1)
            MetaConcrete(bee=1)
            MetaConcrete(2, b=1)  # E: incompatible_call
            MetaConcrete(a=1, bee=1)  # E: incompatible_call
            MetaConcrete(1, 2)  # E: incompatible_call
            model = MetaConcrete(2, bee=1)
            model.b = 3  # E: incompatible_assignment

        def check_inheritance() -> None:
            class FrozenParent(MetaBase, frozen=True):
                x: int

            class MutableChild(FrozenParent, frozen=False):  # E: invalid_base
                y: int

    @assert_passes(run_in_both_module_modes=True)
    def test_dataclass_transform_decorator_field_specifier_options_after_import_failure(
        self,
    ):
        from typing import Any, Callable, TypeVar

        from typing_extensions import dataclass_transform

        T = TypeVar("T")

        def model_field(
            *,
            init: bool = True,
            default: Any = 0,
            alias: str | None = None,
            kw_only: bool = False,
        ) -> Any:
            return object()

        @dataclass_transform(
            kw_only_default=True, frozen_default=True, field_specifiers=(model_field,)
        )
        def create_model(
            *, frozen: bool = True, kw_only: bool = True
        ) -> Callable[[type[T]], type[T]]:
            def decorator(cls: type[T]) -> type[T]:
                return cls

            return decorator

        @create_model()
        class DecoratorModel:
            x: int = model_field(init=False)
            y: int = model_field(alias="why")
            z: int = model_field(default=3, kw_only=False)

        def check_calls() -> None:
            DecoratorModel(2, why=1)
            DecoratorModel(why=1)
            DecoratorModel(2, y=1)  # E: incompatible_call
            DecoratorModel(x=1, why=1)  # E: incompatible_call
            DecoratorModel(1, 2)  # E: incompatible_call
            model = DecoratorModel(2, why=1)
            model.y = 3  # E: incompatible_assignment

    @assert_passes(run_in_both_module_modes=True)
    def test_dataclass_transform_base_field_specifier_options_after_import_failure(
        self,
    ):
        from typing import Any

        from typing_extensions import dataclass_transform

        def model_field(
            *,
            init: bool = True,
            default: Any = 0,
            alias: str | None = None,
            kw_only: bool = False,
        ) -> Any:
            return object()

        @dataclass_transform(
            kw_only_default=True, frozen_default=False, field_specifiers=(model_field,)
        )
        class BaseModel:
            def __init_subclass__(
                cls, *, frozen: bool = False, kw_only: bool = True
            ) -> None:
                pass

        class Concrete(BaseModel, frozen=True):
            a: int = model_field(init=False)
            b: int = model_field(alias="bee")
            c: int = model_field(default=3, kw_only=False)

        def check_calls() -> None:
            Concrete(2, bee=1)
            Concrete(bee=1)
            Concrete(2, b=1)  # E: incompatible_call
            Concrete(a=1, bee=1)  # E: incompatible_call
            Concrete(1, 2)  # E: incompatible_call
            model = Concrete(2, bee=1)
            model.b = 3  # E: incompatible_assignment

    @assert_passes(run_in_both_module_modes=True)
    def test_dataclass_transform_metaclass_field_specifier_options_after_import_failure(
        self,
    ):
        from typing import Any

        from typing_extensions import dataclass_transform

        def model_field(
            *,
            init: bool = True,
            default: Any = 0,
            alias: str | None = None,
            kw_only: bool = False,
        ) -> Any:
            return object()

        @dataclass_transform(
            kw_only_default=True, frozen_default=False, field_specifiers=(model_field,)
        )
        class ModelMeta(type):
            pass

        class MetaBase(metaclass=ModelMeta):
            def __init_subclass__(
                cls, *, frozen: bool = False, kw_only: bool = True
            ) -> None:
                pass

        class MetaConcrete(MetaBase, frozen=True):
            a: int = model_field(init=False)
            b: int = model_field(alias="bee")
            c: int = model_field(default=3, kw_only=False)

        def check_calls() -> None:
            MetaConcrete(2, bee=1)
            MetaConcrete(bee=1)
            MetaConcrete(2, b=1)  # E: incompatible_call
            MetaConcrete(a=1, bee=1)  # E: incompatible_call
            MetaConcrete(1, 2)  # E: incompatible_call
            model = MetaConcrete(2, bee=1)
            model.b = 3  # E: incompatible_assignment

    @assert_passes(allow_import_failures=True)
    def test_dataclass_transform_ignores_inherited_init_after_import_failure(self):
        from typing import Any

        from typing_extensions import dataclass_transform

        def model_field(*, init: bool = True, default: Any = 0) -> Any:
            raise NotImplementedError

        @dataclass_transform(kw_only_default=True, field_specifiers=(model_field,))
        class Base:
            not_a_field: str

            def __init_subclass__(cls, *, kw_only: bool = True) -> None:
                pass

            def __init__(self, not_a_field: str) -> None:
                self.not_a_field = not_a_field

        class Child(Base):
            id: int = model_field()

        Child(id=1)
        Child(not_a_field="x", id=1)  # E: incompatible_call

    @assert_passes()
    def test_dataclass_transform_generic_base(self):
        from typing import Any, Generic, TypeVar

        from typing_extensions import dataclass_transform

        T = TypeVar("T")

        def model_field(*, init: bool = True, default: Any = 0) -> Any:
            return object()

        @dataclass_transform(kw_only_default=True, field_specifiers=(model_field,))
        class GenericBase(Generic[T]):
            data: T

            def __init_subclass__(cls, *, kw_only: bool = True) -> None:
                pass

        class GenericChild(GenericBase[int]):
            id: int = model_field()

        def check_calls() -> None:
            GenericChild(id=1)

    @assert_passes()
    def test_dataclass_transform_order_keyword_controls_comparison(self):
        from typing import Any

        from typing_extensions import dataclass_transform

        def model_field(*, init: bool = True, default: Any = 0) -> Any:
            return object()

        @dataclass_transform(kw_only_default=True, field_specifiers=(model_field,))
        class Base:
            def __init_subclass__(
                cls, *, order: bool = True, kw_only: bool = True
            ) -> None:
                pass

        class Unordered(Base):
            id: int = model_field()

        class Ordered(Base, order=True):
            id: int = model_field()

        def check_calls() -> None:
            unordered1 = Unordered(id=1)
            unordered2 = Unordered(id=2)
            unordered1 < unordered2  # E: unsupported_operation

            ordered1 = Ordered(id=1)
            ordered2 = Ordered(id=2)
            ordered1 < ordered2

    @assert_passes()
    def test_dataclass_transform_hash_semantics(self):
        from dataclasses import dataclass
        from typing import Hashable

        from typing_extensions import dataclass_transform

        @dataclass_transform(eq_default=True, frozen_default=False)
        class Base:
            def __init_subclass__(
                cls, *, eq: bool = True, frozen: bool = False, unsafe_hash: bool = False
            ) -> None:
                dataclass(cls, eq=eq, frozen=frozen, unsafe_hash=unsafe_hash)

        class Unhashable(Base):
            value: int

        class Frozen(Base, frozen=True):
            value: int

        class NoEq(Base, eq=False):
            value: int

        class UnsafeHash(Base, unsafe_hash=True):
            value: int

        class ExplicitHash(Base):
            value: int

            def __hash__(self) -> int:
                return 0

        bad_unhashable: Hashable = Unhashable(value=1)  # E: incompatible_assignment
        ok_frozen: Hashable = Frozen(value=1)
        ok_no_eq: Hashable = NoEq(value=1)
        ok_unsafe_hash: Hashable = UnsafeHash(value=1)
        ok_explicit_hash: Hashable = ExplicitHash(value=1)

    @assert_passes(run_in_both_module_modes=True)
    def test_dataclass_transform_hash_semantics_after_import_failure(self):
        from typing import Hashable

        from typing_extensions import dataclass_transform

        @dataclass_transform(eq_default=True, frozen_default=False)
        class Base:
            def __init_subclass__(
                cls, *, eq: bool = True, frozen: bool = False, unsafe_hash: bool = False
            ) -> None:
                pass

        class Unhashable(Base):
            value: int

        class Frozen(Base, frozen=True):
            value: int

        class NoEq(Base, eq=False):
            value: int

        class UnsafeHash(Base, unsafe_hash=True):
            value: int

        class ExplicitHash(Base):
            value: int

            def __hash__(self) -> int:
                return 0

        def check_hashability() -> None:
            bad_unhashable: Hashable = Unhashable(value=1)  # E: incompatible_assignment
            ok_frozen: Hashable = Frozen(value=1)
            ok_no_eq: Hashable = NoEq(value=1)
            ok_unsafe_hash: Hashable = UnsafeHash(value=1)
            ok_explicit_hash: Hashable = ExplicitHash(value=1)
            print(bad_unhashable, ok_frozen, ok_no_eq, ok_unsafe_hash, ok_explicit_hash)

    @assert_passes(run_in_both_module_modes=True)
    def test_dataclass_transform_init_and_match_args_keywords_after_import_failure(
        self,
    ):
        from typing_extensions import dataclass_transform

        @dataclass_transform()
        class Base:
            def __init_subclass__(
                cls, *, init: bool = True, match_args: bool = True
            ) -> None:
                pass

        class InitDisabled(Base, init=False):
            x: int

        class Matchable(Base, init=False, match_args=True):
            x: int
            y: int

        def check_init_disabled() -> None:
            InitDisabled()
            InitDisabled(1)  # E: incompatible_call

        def allow_positional_patterns(value: Matchable) -> None:
            match value:
                case Matchable(1, 2):
                    pass

        class NoMatchArgs(Base, match_args=False):
            x: int

        def reject_positional_patterns(value: NoMatchArgs) -> None:
            match value:
                case NoMatchArgs(1):  # E: bad_match
                    pass

    @assert_passes(run_in_both_module_modes=True)
    def test_dataclass_transform_factory_field_specifier_after_import_failure(self):
        from typing import Any, Callable, TypeVar

        from typing_extensions import dataclass_transform

        T = TypeVar("T")

        def model_field(
            *,
            init: bool = True,
            default: Any = ...,
            factory: Callable[[], Any] | None = None,
        ) -> Any:
            return object()

        @dataclass_transform(field_specifiers=(model_field,))
        def create_model() -> Callable[[type[T]], type[T]]:
            def decorator(cls: type[T]) -> type[T]:
                return cls

            return decorator

        @create_model()
        class WithFactory:
            x: int = model_field(factory=lambda: 1)

        @create_model()
        class BadFactory:
            x: int = model_field(factory=lambda: "x")  # E: incompatible_assignment

        @create_model()
        class BadOrder:  # E: invalid_dataclass
            x: int = model_field(factory=lambda: 1)
            y: int

        def check_calls() -> None:
            WithFactory()
            WithFactory(1)

    @assert_passes(run_in_both_module_modes=True)
    def test_dataclass_transform_non_default_kwargs_do_not_set_default(self):
        from typing import Any, Callable, TypeVar

        from typing_extensions import dataclass_transform

        T = TypeVar("T")

        def model_field(*, default: Any = ..., repr: bool = True) -> Any:
            return object()

        @dataclass_transform(field_specifiers=(model_field,))
        def create_model() -> Callable[[type[T]], type[T]]:
            def decorator(cls: type[T]) -> type[T]:
                return cls

            return decorator

        @create_model()
        class Model:
            x: int = model_field(repr=False)
            y: int

        def check_calls() -> None:
            Model(1, 2)
            Model(1)  # E: incompatible_call

    @assert_passes()
    def test_dataclass_transform_converter_field_specifier(self):
        from typing import Callable, TypeVar

        from typing_extensions import assert_type, dataclass_transform

        T = TypeVar("T")
        S = TypeVar("S")

        def model_field(
            *,
            converter: Callable[[S], T],
            default: S | None = None,
            default_factory: Callable[[], S] | None = None,
        ) -> T:
            return object()

        @dataclass_transform(field_specifiers=(model_field,))
        class Base:
            pass

        def to_int(value: str) -> int:
            return int(value)

        class ConverterClass:
            def __init__(self, value: str | bytes) -> None:
                pass

        class Model(Base):
            value: int = model_field(converter=to_int)
            class_value: ConverterClass = model_field(converter=ConverterClass)
            converted_default: int = model_field(converter=to_int, default_factory=str)

            # E: incompatible_call
            bad_default: int = model_field(converter=to_int, default_factory=int)

        def check_calls() -> None:
            model = Model("1", b"3", "2")
            assert_type(model.value, int)
            assert_type(model.class_value, ConverterClass)

            model.value = "4"
            model.class_value = "5"
            model.class_value = b"6"

            model.value = 4  # E: incompatible_assignment
            model.class_value = 7  # E: incompatible_assignment

            Model(1, b"3", "2")  # E: incompatible_argument
            Model("1", 2, "3")  # E: incompatible_argument
            Model("1", b"3", 2)  # E: incompatible_argument

    @assert_passes(allow_import_failures=True)
    def test_dataclass_transform_converter_dict(self):
        from typing import Callable, TypeVar

        from typing_extensions import assert_type, dataclass_transform

        T = TypeVar("T")
        S = TypeVar("S")

        def model_field(*, converter: Callable[[S], T], default: S) -> T:
            raise NotImplementedError

        @dataclass_transform(field_specifiers=(model_field,))
        class Base:
            pass

        class Model(Base):
            mapping_value: dict[str, str] = model_field(converter=dict, default=())

        def capybara():
            model = Model(())
            assert_type(model.mapping_value, dict[str, str])

    @assert_passes(allow_import_failures=True)
    def test_dataclass_transform_converter_field_specifier_after_import_failure(self):
        from typing import Callable, TypeVar

        from typing_extensions import assert_type, dataclass_transform

        T = TypeVar("T")
        S = TypeVar("S")

        def model_field(
            *,
            converter: Callable[[S], T],
            default: S | None = None,
            default_factory: Callable[[], S] | None = None,
        ) -> T:
            raise NotImplementedError

        @dataclass_transform(field_specifiers=(model_field,))
        class Base:
            pass

        def to_int(value: str) -> int:
            return int(value)

        class ConverterClass:
            def __init__(self, value: str | bytes) -> None:
                pass

        class Model(Base):
            value: int = model_field(converter=to_int)
            class_value: ConverterClass = model_field(converter=ConverterClass)
            converted_default: int = model_field(converter=to_int, default_factory=str)
            mapping_value: dict[str, str] = model_field(converter=dict, default=())

            # E: incompatible_call
            bad_default: int = model_field(converter=to_int, default_factory=int)

        def check_calls() -> None:
            model = Model("1", b"3", "2", (("a", "1"),))
            assert_type(model.value, int)
            assert_type(model.class_value, ConverterClass)
            assert_type(model.mapping_value, dict[str, str])

            model.value = "4"
            model.class_value = "5"
            model.class_value = b"6"
            model.mapping_value = {"b": "2"}
            model.mapping_value = (("c", "3"),)

            model.value = 4  # E: incompatible_assignment
            model.class_value = 7  # E: incompatible_assignment

            Model(1, b"3", "2", ())  # E: incompatible_argument
            Model("1", 2, "3", ())  # E: incompatible_argument
            Model("1", b"3", 2, ())  # E: incompatible_argument

    @assert_passes(allow_import_failures=True)
    def test_dataclass_transform_descriptor_fields_after_import_failure(self):
        from typing import Any, Generic, TypeVar, assert_type, overload

        from typing_extensions import dataclass_transform

        T = TypeVar("T")

        class DataDescriptor:
            @overload
            def __get__(self, instance: None, owner: Any) -> "DataDescriptor": ...

            @overload
            def __get__(self, instance: object, owner: Any) -> int: ...

            def __get__(
                self, instance: object | None, owner: Any
            ) -> "DataDescriptor | int":
                raise NotImplementedError

            def __set__(self, instance: object, value: int) -> None:
                raise NotImplementedError

        class ReadDescriptor(Generic[T]):
            @overload
            def __get__(self, instance: None, owner: Any) -> list[T]: ...

            @overload
            def __get__(self, instance: object, owner: Any) -> T: ...

            def __get__(self, instance: object | None, owner: Any) -> list[T] | T:
                raise NotImplementedError

        @dataclass_transform()
        def model(cls: type[T]) -> type[T]:
            return cls

        @model
        class TransformModel:
            b: ReadDescriptor[int]
            a: DataDescriptor = DataDescriptor()
            c: ReadDescriptor[str] = ReadDescriptor()

        assert_type(TransformModel.a, DataDescriptor)
        assert_type(TransformModel.b, list[int])
        assert_type(TransformModel.c, list[str])

        transformed = TransformModel(ReadDescriptor(), 1, ReadDescriptor())
        assert_type(transformed.a, int)
        assert_type(transformed.b, int)
        assert_type(transformed.c, str)

        # E: incompatible_argument
        TransformModel(ReadDescriptor(), DataDescriptor(), ReadDescriptor())
        TransformModel(1, 2, 3)  # E: incompatible_argument # E: incompatible_argument
