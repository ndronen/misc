#!/bin/bash

sudo groupadd --gid 1000 ubuntu
sudo useradd -g ubuntu -c "Ubuntu compatibility" -s /bin/bash --uid 1000 -m ubuntu
sudo cp -r /home/ec2-user/.ssh /home/ubuntu
sudo chown -R ubuntu:ubuntu /home/ubuntu/.ssh
sudo visudo
sudo mkdir /home/ubuntu/proj
if ! grep -q xvdf /etc/fstab 
then
    echo -e  "/dev/xvdf\t/home/ubuntu/proj\text4\tdefaults,noatime\t1\t1" | sudo tee -a /etc/fstab
fi
sudo mount -a
