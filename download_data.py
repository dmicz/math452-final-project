import urllib.request
import tarfile

# URL to download NIH data
link = 'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz'
urllib.request.urlretrieve(link, 'images.tar.gz')

# extract images
file_name = 'images.tar.gz'  # Replace with your file name
with tarfile.open(file_name, 'r:gz') as tar:
    tar.extractall() 