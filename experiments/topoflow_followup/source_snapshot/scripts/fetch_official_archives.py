"""Fetch public institutional source archives without depending on the legacy API."""
import hashlib
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parents[1]
ARCHIVES=[('https://imperialcollegelondon.box.com/shared/static/2x0zkaqw9swr47nqjm9xww6z9iu39sn0.zip',
           ROOT/'data/public/ImagesICPSC2007.zip','d07c1d3403a19d4b39492d9671805cfd3b65c2f5'),
          ('https://imperialcollegelondon.box.com/shared/static/2n4lb95amz7fstk19q826uzu922eb4b1.zip',
           ROOT/'data/public/estaillades/Estaillades_1000c_3p31136um.zip',None)]


def main():
    for url,path,expected in ARCHIVES:
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            partial=path.with_suffix(path.suffix+'.part')
            with httpx.stream('GET',url,follow_redirects=True,timeout=120) as response:
                response.raise_for_status()
                with partial.open('wb') as output:
                    for block in response.iter_bytes():output.write(block)
            partial.replace(path)
        if expected and hashlib.sha1(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Archive checksum mismatch: {path.name}')
        print(f'Available official archive: {path.name}',flush=True)


if __name__=='__main__':main()
