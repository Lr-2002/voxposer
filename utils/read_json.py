import json
import gzip


with gzip.open("/home/yue/桌面/ovmm/home-robot/data/datasets/ovmm/train/episodes.json.gz", "rb") as file:
    data = json.loads(file.read())
    
def show_hierarchy(level, cur):
    if isinstance(cur, dict):
        print(" "*4*level+'{')
        file.write(" "*4*level+'{'+'\n')
        for k in cur:
            print(" "*4*level + k)
            file.write(" "*4*level + k + '\n')
            show_hierarchy(level+1, cur[k])
        print(" "*4*level+'}')
        file.write(" "*4*level+'}'+ '\n')
    elif isinstance(cur, list):
        print(" "*4*level+'[')
        file.write(" "*4*level+'['+ '\n')
        if len(cur) != 0:
            show_hierarchy(level+1, cur[0])
        print(" "*4*level+'...')
        
        file.write(" "*4*level+'...'+ '\n')
        print(" "*4*level+']')
        file.write(" "*4*level+']'+ '\n')


if __name__ == '__main__':
    show_hierarchy(0, data)
    print(data['episodes'][0])