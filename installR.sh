#!/bin/bash

#sudo apt-get --purge remove -y r-base-core
codename=$(lsb_release -c -s)
echo "deb http://cran.cnr.berkeley.edu/bin/linux/ubuntu $codename/" | sudo tee -a /etc/apt/sources.list > /dev/null
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
sudo add-apt-repository -y ppa:marutter/rdev
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y r-base=3.4.0-1xenial0




#sudo apt-get install r-base=3.4.0-1precise0
#sudo apt-get install gdebi-core
#wget https://download1.rstudio.org/rstudio-1.0.44-amd64.deb
#sudo gdebi -n rstudio-1.0.44-amd64.deb
#rm rstudio-1.0.44-amd64.deb
