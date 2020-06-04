Дипломная работа
================
Дипломная работа в LaTeX на тему «Решение прямой задачи бокового каротажного зондирования методами численного моделирования».

Структура исходников
--------------------
Структура каталогов:
```
.
├── images
└── inc
```

В корневом каталоге `.` находится файл `main.tex`.
В `main.tex` подключается все, включая преамбулу, титульный лист, приложения и т.д.

В каталоге `images/` находятся рисунки, схемы.

В каталоге `inc/` находятся файлы, которые подключаются к `main.tex`:
* файлы формата `0-*.tex` являются ненумерованными секциями (например введение, заключение, список использованной литературы)
* файлы формата `[1-9]-*.tex` являются нумерованными секциями (например постановка задачи и т.д)

Работа с LaTeX
--------------
Как установить LaTeX: http://blog.amet13.name/2014/06/latex.html

Пример компиляции проекта с помощью Makefile:
```bash
make
```

Пример очистки сборочных файлов после компиляции (кроме PDF):
```bash
make clean
```

Благодарности
-------------
[Амету Умерову](https://github.com/Amet13/bachelor-diploma/) за основу шаблона LaTeX