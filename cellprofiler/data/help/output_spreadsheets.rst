Using Spreadsheets and Databases
================================

CellProfiler can save measurements as a *spreadsheet* or as a *database*.
Which format you use will depend on some of the considerations below:

-  *Learning curve:* Applications that handle spreadsheets (e.g., Excel,
   `Calc`_ or `Google Docs`_) are easy for beginners to use. Databases
   are more sophisticated and require knowledge of specialized languages
   (e.g., MySQL, Oracle, etc); a popular freeware access tool is
   `SQLyog`_.
-  *Capacity and speed:* Databases are designed to hold larger amounts
   of data than spreadsheets. Spreadsheets may contain a few
   thousand rows of data, whereas databases can hold many millions of
   rows of data. Accessing a particular portion of data in a database
   is optimized for speed.
-  *Downstream application:* If you wish to use Excel or another simple
   tool to analyze your data, a spreadsheet is likely the best choice.  If you
   intend to use CellProfiler Analyst, you must create a database.  If you
   plan to use a scripting language, most languages have ways to import
   data from either format.

.. _Calc: http://www.libreoffice.org/discover/calc/
.. _Google Docs: http://docs.google.com
.. _SQLyog: http://www.webyog.com/