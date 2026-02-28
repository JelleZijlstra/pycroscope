# static analysis: ignore

from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestDataclassTransform(TestNameCheckVisitorBase):
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
        boom = 1 / 0

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
        boom = 1 / 0

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
        class BadModel:  # E: invalid_annotation
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
            decorated = Decorated(2, why=1)
            decorated.y = 4  # E: incompatible_assignment

            FromBase(2, bee=1)
            FromBase(2, b=1)  # E: incompatible_call
            from_base = FromBase(2, bee=1)
            from_base.b = 4  # E: incompatible_assignment

            FromMeta(2, enn=1)
            FromMeta(2, n=1)  # E: incompatible_call
            from_meta = FromMeta(2, enn=1)
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

    @assert_passes(allow_import_failures=True)
    def test_dataclass_transform_decorator_field_specifier_options_after_import_failure(
        self,
    ):
        boom = 1 / 0

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
        class DecoratorModel:
            x: int = model_field(init=False)
            y: int = model_field(alias="why")
            z: int = model_field(default=3, kw_only=False)

        DecoratorModel(2, why=1)
        DecoratorModel(why=1)
        DecoratorModel(2, y=1)  # E: incompatible_call
        DecoratorModel(x=1, why=1)  # E: incompatible_call
        DecoratorModel(1, 2)  # E: incompatible_call
        model = DecoratorModel(2, why=1)
        model.y = 3  # E: incompatible_assignment

    @assert_passes(allow_import_failures=True)
    def test_dataclass_transform_base_field_specifier_options_after_import_failure(
        self,
    ):
        boom = 1 / 0

        from typing import Any

        from typing_extensions import dataclass_transform

        def model_field(
            *,
            init: bool = True,
            default: Any = 0,
            alias: str | None = None,
            kw_only: bool = False,
        ) -> Any:
            raise NotImplementedError

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

        Concrete(2, bee=1)
        Concrete(bee=1)
        Concrete(2, b=1)  # E: incompatible_call
        Concrete(a=1, bee=1)  # E: incompatible_call
        Concrete(1, 2)  # E: incompatible_call
        model = Concrete(2, bee=1)
        model.b = 3  # E: incompatible_assignment

    @assert_passes(allow_import_failures=True)
    def test_dataclass_transform_metaclass_field_specifier_options_after_import_failure(
        self,
    ):
        boom = 1 / 0

        from typing import Any

        from typing_extensions import dataclass_transform

        def model_field(
            *,
            init: bool = True,
            default: Any = 0,
            alias: str | None = None,
            kw_only: bool = False,
        ) -> Any:
            raise NotImplementedError

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

        MetaConcrete(2, bee=1)
        MetaConcrete(bee=1)
        MetaConcrete(2, b=1)  # E: incompatible_call
        MetaConcrete(a=1, bee=1)  # E: incompatible_call
        MetaConcrete(1, 2)  # E: incompatible_call
        model = MetaConcrete(2, bee=1)
        model.b = 3  # E: incompatible_assignment

    @assert_passes(allow_import_failures=True)
    def test_dataclass_transform_ignores_inherited_init_after_import_failure(self):
        boom = 1 / 0

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

    @assert_passes(allow_import_failures=True)
    def test_dataclass_transform_hash_semantics_after_import_failure(self):
        boom = 1 / 0

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

        bad_unhashable: Hashable = Unhashable(value=1)  # E: incompatible_assignment
        ok_frozen: Hashable = Frozen(value=1)
        ok_no_eq: Hashable = NoEq(value=1)
        ok_unsafe_hash: Hashable = UnsafeHash(value=1)
        ok_explicit_hash: Hashable = ExplicitHash(value=1)
