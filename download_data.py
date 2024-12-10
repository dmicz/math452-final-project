import urllib.request
import tarfile
import os
from tqdm import tqdm
import pandas as pd

# # all NIH data links
# links = [
#     'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
#     'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
#     'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
# 	'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
#     'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
# 	'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
# 	'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
#     'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
# 	'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
# 	'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
# 	'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
# 	'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
# ]

# # loop through links to download
# i = 0
# for link in tqdm(links, desc = 'loading files'):

#     file_name = f'images_{i}.tar.gz'

#     # URL to download NIH data
#     urllib.request.urlretrieve(link, file_name)

#     # extract images
#     file_name = f'images_{i}.tar.gz'
#     with tarfile.open(file_name, 'r:gz') as tar:
#         tar.extractall() 

#     # get rid of excess tar files
#     os.remove(file_name)

#     i+=1

# CSV containing image labels
img_data = pd.read_csv('Data_Entry_2017_v2020.csv')

# find most frequent labels (attempt to balance data)
most_freq = set(list(img_data['Finding Labels'].value_counts()[:10].index))

# pick 1300 images from each label
valid_files = set()
for label in most_freq:
    label_images = img_data[(img_data['Finding Labels'] == label) & (img_data['Finding Labels'] != 'No Finding')]
    selected_images = label_images.head(1300)['Image Index']
    valid_files.update(selected_images)

# remove other files
image_files = set(os.listdir('images'))
files_to_remove = image_files - valid_files

for file in files_to_remove:
    os.remove(f'images/{file}')