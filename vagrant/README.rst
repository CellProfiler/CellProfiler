# metabolism

## Getting started

1. Download and install [Vagrant](https://www.vagrantup.com/downloads.html).

1. Download and install [VirtualBox](https://www.virtualbox.org/wiki/Downloads).

1. Clone this repository:
    ```sh
    git clone git@github.com:CellProfiler/metabolism.git
    ```

## Usage

Install Virtualbox (https://www.virtualbox.org/wiki/Downloads)

Then:
```sh
cd metabolism
vagrant up
```

Clone CP and CPA:
```sh
git clone https://github.com/CellProfiler/CellProfiler
git clone https://github.com/CellProfiler/CellProfiler-Analyst
```

SSH into virtual machine
```sh
vagrant ssh
```

Start CellProfiler Analyst
```sh
python /vagrant/CellProfiler-Analyst/CellProfiler-Analyst.py
```
