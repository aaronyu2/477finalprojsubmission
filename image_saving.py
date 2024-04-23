import os

def save_file_folder(contents,
                     basepath: str,
                     filename: str,
                     depth:int =4) -> str:
    temp_folder = os.path.join(*filename[:depth])
    folder = os.path.join(basepath, temp_folder)
    os.makedirs(folder, exist_ok=True)
    if type(contents) == str:
        with open(os.path.join(folder, filename), 'w') as f:
            f.write(contents)
    else:
        with open(os.path.join(folder, filename), 'wb') as f:
            f.write(contents)
    return str(os.path.join(folder, filename))
  

if __name__=='__main__':
    contents = 'Hello World'
    basepath = '/home/ubuntu/test'
    filename = 'test.txt'
    print(save_file_folder(contents, basepath, filename))