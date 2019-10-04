import os
from tqdm import tqdm
from PIL import Image

def compress_image(infile, outfile='',quality=10):
    im = Image.open(infile)
    if 'A' in im.getbands():
        return
    im = im.convert('RGB')
    w, h = im.size
    ratio = 480/max(w, h)
    if(ratio<1):
        im = im.resize((int(w*ratio), int(h*ratio)), Image.BICUBIC)
    im.save(outfile, quality=quality)


if not os.path.exists("x"):
    os.mkdir("x")
if not os.path.exists("y"):
    os.mkdir("y")
tasks = os.listdir("ori")
for t in tqdm(tasks):
    t_name = '.'.join(t.split('.')[:-1])
    compress_image(
        os.path.join("ori", t),
        os.path.join("x", t_name+'.jpg'),    
        quality=100,    
    )
    compress_image(
        os.path.join("ori", t),
        os.path.join("y", t_name+'.jpg'),        
    )
