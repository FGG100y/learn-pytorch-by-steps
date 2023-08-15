"""
Python Class review.


## Classes vs Instances
Classes are used to create user-defined data structures. Classes define
functions called *methods*, which identify the behaviours and actions that an
object created from the class can perform with its data.

A class is a blueprint for how something should be defined. It doesn't actually
contain any data.

While the class is the blueprint, an *instance* is an object that is built from
a class and contains real data.

A class' instance object are mutable by default. An object is mutable if it can
be altered dynamically (such as lists).

## Inherit from other classes: (Is-a relationship)
Inheritance is the process by which one class takes on the attributes and
methods of another.

    - Newly formed classes are called "child classes", "derived classes",
    "subclasses", or "subtypes";

    - and the classes that child classes are derived from are called "parent
      classes", "base classes", or "super classes".

    - A derived class is said to derive, inherit, or extend a base class.

>Liskov substitution principle:
    "In a computer program, if S is a subtype of T, then objects of type T may
    be replaced with objects of type S without altering any of the desired
    properties of the program."

## Composition of classes: (Has-a relationship)
Composition is a concept that models a "has-a" relationship. It enables
creating complex types by combining objects of other types.
A class "composite" can contain an object of another class "component", i.e.,
a composite has a component.

Composition allows composite classes to reuse the implementation of the
components it contains. The composite class doesn't inherit the component class
interface, but it can leverage its implementation.

The composition relation between two classes is considered loosely coupled.
That means that changes to the component class rarely affect the composite
class, and changes to the composite class never affect the component class.

This provides better adaptability to change and allows applications to
introduce new requirements without affecting existing code.

(lack of class Computes example)
"""

# The Dog Park


class Dog:

    # class attribute: (for all instances)
    species = "Chinese countrysied"

    # ----------------
    # dunder methods (double underscores methods)
    # ----------------
    # .__init__() initialize each new instance of the class
    # its first parameter will always be a variable named 'self'
    # when a new class instance is created, the instance is automatically
    # passed to the 'self' parameter so that new attributes can be attached
    def __init__(self, name, age, breed):
        # instance attributes: (for the instance only)
        self.name = name
        self.age = age
        self.breed = breed

    def __str__(self):
        return f"{self.name} is {self.age} years old."

    # ----------------
    # instance methods
    # ----------------
    #  # a valid method the description() is, but not so "pythonic"
    #  # especially when using the print(Dog("erwang", 8)); using __str__()!
    #  def description(self):
    #      return f"{self.name} is {self.age} years old."
    def speak(self, sound="wolf"):
        return f"{self.name} barks: {sound}"


class TianYuan01(Dog):
    def speak(self, sound="wang-wang"):
        # partially override speak() method
        # access the parent class by using super()
        # super() returns a temporary object of the parent-class that then
        # allows child-class to call that parent-class' method:
        return super().speak(sound)


class ErLangShenQuan(Dog):
    def speak(self, sound="你瞅啥"):
        # completely override speak() method
        return f"{self.name} says: {sound}"


# The Geometric World


class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def __str__(self):
        return f"Rectangle with {self.length=}, {self.width=}"

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * (self.length + self.width)


# super() in single Inheritance:
class Square(Rectangle):
    def __init__(self, length):
        # super() can take two parameters: subclass and instance of subclass
        #  super(Square, self).__init__(length, length)

        # or not so verbose and the RECOMMANDED way:
        super().__init__(length, length)

    def __str__(self):
        return f"Square with {self.length=}"


# extend functionality of .area()
class Cube(Square):

    def __str__(self):
        return super().__str__()

    def surface_area(self):
        face_area = super().area()
        return 6 * face_area

    def volume(self):
        face_area = super().area()
        return face_area * self.length

    # if Square also impelemented an .area() method and you wanted to make sure
    # that Cube did not use:
    #
    # setting Square as the subclass argument to super() instead of Cude,
    # this cause super() to start searching for a matching method (.area()) at
    # one level above Square in the instance hierarchy (i.e., the Rectangle)
    #
    #  def surface_area(self):
    #      face_area = super(Square, self).area()
    #      return 6 * face_area
    #
    #  def volume(self):
    #      face_area = super(Square, self).area()
    #      return face_area * self.length


# Multiple Inheritance:
#  A subclass can inherit from multiple superclasses that
#  don't necessarily inherit from each other (aka sibling classes)
# super() in Multiple Inheritance

class Triangle:
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def __str__(self):
        return f"Triange with {self.base=}, {self.height=}"

    def tri_area(self):
        return 0.5 * (self.base * self.height)


#  print(RightPyramid.__mro__) to check the Method-Resolution-Order
class RightPyramid(Square, Triangle):  # the signature order matters!
    """A pyramid with a square base
    """
    def __init__(self, base, slant_height):
        self.base = base
        self.slant_height = slant_height
        # initialize partially with .__init__() from Square class
        super().__init__(self.base)

    def __str__(self):
        return f"RightPyramid with {self.base=}, {self.slant_height=}"

    def area(self):
        base_area = super().area()
        perimeter = super().perimeter()
        return 0.5 * (perimeter * self.slant_height) + base_area

    def area_2(self):
        base_area = super().area()
        triangle_area = super().tri_area()
        return 4 * triangle_area + base_area


# NOTE the above Multiple Inheritance examlpe is not so good. Keep learning.


if __name__ == "__main__":
    d0 = Dog(name='wqai', age=3, breed='zhty-01')
    d1 = TianYuan01(name='fqai', age=4, breed='zhty-02')
    d2 = ErLangShenQuan(name='erha', age=2, breed='shenq-01')
    for dog in (d0, d1, d2):
        print(dog, dog.speak())

    ret = Rectangle(2, 5)
    squ = Square(3)
    tri = Triangle(4, 2)
    pyramid = RightPyramid(2, 4)
    for shape in (ret, squ, tri, pyramid):
        if isinstance(shape, Triangle):
            print(shape, "whose area =", shape.tri_area())
        elif isinstance(shape, RightPyramid):
            print(shape, "whose area =", shape.area(), shape.area_2())
        else:
            print(shape, "whose area =", shape.area())
