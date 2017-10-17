macOS
=====

Prerequisites
-------------

Homebrew
~~~~~~~~

`Homebrew`_ is a free and open-source software package management system
that simplifies the installation of software on Apple’s macOS operating
system.

Installation
^^^^^^^^^^^^

.. code:: sh

    $ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Java
~~~~

`Java`_ is a general-purpose computer programming language that is concurrent, 
class-based, object-oriented, and specifically designed to have as few 
implementation dependencies as possible.

Installation
^^^^^^^^^^^^

.. code:: sh

    $ brew cask install java

MySQL
~~~~~

`MySQL`_ is an open-source relational database management system (RDBMS).

Installation
^^^^^^^^^^^^

.. code:: sh

    $ brew install mysql

Python
~~~~~~

`Python`_ is a widely used high-level programming language for
general-purpose programming, created by Guido van Rossum and first
released in 1991.

Installation
^^^^^^^^^^^^

.. code:: sh

    $ brew install python

UPX
~~~

`UPX`_ (Ultimate Packer for Executables) is a free and open source
executable packer supporting a number of file formats from different
operating systems.

Installation
^^^^^^^^^^^^

.. code:: sh

    $ brew install upx

wxPython
~~~~~~~~

`wxPython`_ is a wrapper for the cross-platform GUI API (often referred to as 
a “toolkit”) `wxWidgets`_ (which is written in C++) for the Python programming 
language.

Installation
^^^^^^^^^^^^

.. code:: sh

    $ brew install wxmac --build-from-source --with-static --with-stl

    $ brew install wxpython --build-from-source

Use
---

Use the Makefile to create an `Apple Disk Image`_ (CellProfiler.dmg)
that contains the macOS `package`_ (CellProfiler.app):

.. code:: sh

    $ make

.. _Apple Disk Image: https://en.wikipedia.org/wiki/Apple_Disk_Image
.. _Homebrew: https://brew.sh
.. _Java: https://java.com
.. _MySQL: https://www.mysql.com
.. _package: https://en.wikipedia.org/wiki/Package_(macOS)
.. _Python: https://en.wikipedia.org/wiki/Python_(programming_language)
.. _UPX: https://upx.github.io
.. _wxPython: https://wxpython.org
.. _wxWidgets: https://wxwidgets.org
