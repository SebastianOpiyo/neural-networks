import json
import collections

# Json has the ability to take bumpy objects and serializing them.
# Bumpy objects include a tuple within a list, an nparray, etc.

Jabberwocky = ["’Twas brillig, and the slithy toves",
               "Did gyre and gimble in the wade;",
               "All mimsy were the borogoves,",
               "And the mome raths outgrabe.",
               "",
               "'Beware the Jabberwock, my son!",
               "The jaws that bite, the claws that catch!",
               "Beware the Jubjub bird, and shun",
               "The frumious Bandersnatch!'",
               "",
               "He took his vorpal sword in hand:",
               "Long time the manxome foe he sought—",
               "So rested he by the Tumtum tree,",
               "And stood awhile in thought.",
               "",
               "And as in uffish thought he stood,",
               "The Jabberwock, with eyes of flame,",
               "Came whiffling through the tulgey wood,",
               "And burbled as it came!",
               "",
               "One, two! One, two! And through and through",
               "The vorpal blade went snicker-snack!",
               "He left it dead, and with its head",
               "He went galumphing back.",
               "",
               "'And hast thou slain the Jabberwock?",
               "Come to my arms, my beamish boy!",
               "O frabjous day! Callooh! Callay!'",
               "He chortled in his joy.",
               "",
               "’Twas brillig, and the slithy toves",
               "Did gyre and gimble in the wabe;",
               "All mimsy were the borogoves,",
               "And the mome raths outgrabe."
               ]

# We are using the context manager "with" so we don't need to close the file.
# When the program gets to end of the file it calls the exit func

"""Working without json.
- has limitation especially when the data becomes really bumpy"""


# the data structure is a real simple one
def without_json():
    with open("Jab", "a") as f:
        for line in Jabberwocky:
            f.write(line + "\n")

    with open("Jab", "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            print(line[:-1])


flat = "I am sebastian"
bumpy = ["Apple is sweet", "Did you try an Orange"]
really_bumpy = ["A pancake", [2, 5, "A sausage"]]
badly_bumpy = ["A pancake", ('a', 'azonto'), {'name': "Sebastiana",
                                              3: "Deportivo", 8: "Man City"}, [2, 5, "A sausage"]]
deque_data = collections.deque(badly_bumpy)


# More bumpy data presented in here.
# We are able to overcome most of the shortcomings with the use of json.
def with_json():
    with open("data.txt", "w") as f:
        # this will fail with a deque.
        # because type deque is not json serializable.
        # we need to create json hooks to solve that problem.
        json.dump(deque_data, f)

    # with open("data.txt", "r") as f:
    #     my_obj = json.load(f)
    #     print(type(my_obj))
    #     print(my_obj)


"""A more complicated example with class objects encoding and decoding."""


class Dog:
    def __init__(self, name, age, tricks):
        self._name = name
        self._age = age
        self._tricks = tricks

    def __str__(self):
        ret_str = f"{self._name} is {self._age} years old and knows"
        ret_str += f"the following tricks. \n"
        for trick in self._tricks:
            ret_str += f"{trick} \n"
        return ret_str


fido = Dog("Fido", 10, ['sit', 'stay', 'roll over', 'Jump'])
# print(fido)
# returns the object we need.
# print(fido.__dict__)


class MultiTypeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, collections.deque):
            return {"__deque__": list(o)}
        elif isinstance(o, Dog):
            return {"__Dog__": o.__dict__}
        else:
            json.JSONEncoder.default(o)


def multi_type_decoder(o):
    if "__deque__" in o:
        return collections.deque(o["__deque__"])
    if "__Dog__" in o:
        dec_obj = o["__Dog__"]
        name = dec_obj["_name"]
        age = dec_obj["_age"]
        tricks = list(dec_obj["_tricks"])
        ret_obj = Dog(name, age, tricks)
        return ret_obj
    else:
        return o


with open("data.txt", "w") as f:
    # We add the json hook to enable it works.
    json.dump(fido, f, cls=MultiTypeEncoder)

# with open("data.txt", "r") as f:
#     my_obj = json.load(f, object_hook=multi_type_decoder)
#     print(type(my_obj))
#     print(my_obj)


# if __name__ == '__main__':
#     without_json()
#     # with_json()
