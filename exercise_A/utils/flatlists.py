# We can also use itertools module.
# Since it will be nice if we take the abstraction away, it is what is is.
# import itertools
import more_itertools


class FlattenedIterator:
    """A base class that performs iteration on an iterable data structure."""
    def __init__(self, data):
        self.iterable_list = data
        self.flat_list = []

    # def flatten_approach_one(self):
    #     # Approach one
    #     for iterable in self.iterable_list:
    #         for item in iterable:
    #             self.flat_list.append(item)
    #     print(self.flat_list)
    #     # Comment out the line below if you want to return.
    #     # return self.flat_list

    def flatten_alternative_approach(self):
        # A simplified Alternative approach using list compression.
        # Expects lists with equal dimension otherwise zip() function drops the odd elements
        self.flat_list = [item for iterable in zip(*self.iterable_list) for item in iterable]
        print(self.flat_list)
        # Or if needed to return then comment out the line below
        # return self.flat_list

    def flatten_iterleave_approach(self):
        # This works fine...
        self.flat_list = list(more_itertools.interleave_longest(*self.iterable_list))
        print(self.flat_list)
        # Or if needed to return then comment out the line below
        # return self.flat_list

    """NOTE: there exists other methods, especially with use of 
    modules."""


def _main():
    """This is for instantiating the class and testing the methods in the factory below...
    Nothing clever."""
    list_to_flatten = [[1, 2, 3], [4, 5], [6, 7, 8]]
    test_approach_one = FlattenedIterator(list_to_flatten)
    test_approach_one.flatten_iterleave_approach()
    # test_approach_one.flatten_alternative_approach()


if __name__ == "__main__":
    _main()
