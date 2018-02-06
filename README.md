# README #

This is a simple project in java implementing a classifier based in text descriptors stracteds from pfd files.
This project use weka library to predict new pdf docs and realise traine.

you must use a initial simple structure to run and classify docs.

1 - create inside res/traineset new folders representing the pdf classes.
Ex: res/traineset/to_pay
    res/traineset/payd
2 - Put at folder res/toclassify, new pdfs to predict class. After predict class the docs will moved to correct folder
inside res/classifieds
