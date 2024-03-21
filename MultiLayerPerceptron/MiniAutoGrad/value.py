import torch

class Value:
    def __init__(self, data:torch.float, _children=(), _op='') -> None:
        """
        Initializes a new instance of the class with the given data.

        Args:
            data (torch.float): The data to be stored in the instance.
            _children (tuple): To keep track of any children which resultant the value object.
            _op (str): To keep track of the operation which resulted in the value object.
        """
        self.data = data
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        """
        Return a string representation of the object.
        """
        return f"Value(data={self.data})"

    def __add__(self, other:torch.float):
        """
        Adds the data of the current Value object with the data of another torch.float object.
        Support the + symbol for addition.

        Parameters:
            other (torch.float): The torch.float object to be added to the current Value object.

        Returns:
            Value: A new Value object with the sum of the data from the current Value object.
        """
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other:torch.float):
        """
        Multiply the current value by the given number.
        Support the * symbol for addition.

        Args:
            other (torch.float): The number to multiply the current value by.

        Returns:
            Value: A new Value object representing the result of the multiplication.
        """
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(self.data * other.data, (self, other), '*')
        return out


if __name__ == '__main__':
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    d = a * 5
    e = c+d
    print(c.data)
    print(d.data)
    print(e._prev)
    print(e._op)
