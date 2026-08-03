from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    email: str  

new_person: Person = {
    "name": "John Doe",
    "age": 30,
    "email": "hello@xyz.com"}

print(new_person)