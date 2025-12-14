import sys
from zadanie13 import zad_nie13
from zadanie7 import zad_anie7

def main():
    print("Выберите задание:")
    print("1 - Задание 13 (градиенты, loss, accuracy)")
    print("2 - Задание 7 (инициализации HeNormal vs GlorotNormal)")
    
    choice = input("Введите номер 1 или 2: ").strip()
    
    if choice == "1":
        zad_nie13()
    elif choice == "2":
        zad_anie7()
    else:
        print("Ошибка!")

if __name__ == "__main__":
    main()
