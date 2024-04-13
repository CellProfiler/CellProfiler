# MacOS

## Prerequisites

### Homebrew

[Homebrew](https://brew.sh) is a free and open-source software package management system that simplifies the installation of software on Apple’s macOS operating system.

#### Installation

```sh
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

### Java

[Java](https://java.com) is a general-purpose computer programming language that is concurrent, class-based, object-oriented, and specifically designed to have as few implementation dependencies as possible.

#### Installation

```sh
brew cask install java
```

### MySQL

[MySQL](https://www.mysql.com) is an open-source relational database management system (RDBMS).

#### Installation

```sh
brew install mysql
```

### Python

[Python](https://en.wikipedia.org/wiki/Python_(programming_language)) is a widely used high-level programming language for general-purpose programming, created by Guido van Rossum and first released in 1991.

#### Installation

```sh
brew install python@3.9
```

### UPX

[UPX](https://upx.github.io) (Ultimate Packer for Executables) is a free and open source executable packer supporting a number of file formats from different operating systems.

#### Installation

```sh
brew install upx
```

## Use

Use the Makefile to create an [Apple Disk Image](https://en.wikipedia.org/wiki/Apple_Disk_Image) (CellProfiler.dmg) that contains the MacOS [package](https://en.wikipedia.org/wiki/Package_(macOS)) (CellProfiler.app), making sure to specify the version via environment variable `CP_VERSION`:

```sh
# Specify the version via environment variable CP_VERSION
# can set manually if different version desired
# can exclude --strip-dev for dev/local parts of version string
export CP_VERSION="$(python -m setuptools_scm --strip-dev -c ../../src/frontend/pyproject.toml)"

make CP_VERSION="$CP_VERSION"
```

Among other things, the Makefile constructs the file `Info.plist` from `Info.plist.template`. The `Info.plist.template` file contains the version number in double curly braces, `{{CP_VERSION}}`, which in `Info.plist` is replaced by the actual version you specify above.
