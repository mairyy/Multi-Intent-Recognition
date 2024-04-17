/bin/mkdir -p data/conceptnet

cd data/conceptnet

/bin/wget https://ttic.uchicago.edu/~kgimpel/comsense_resources/train100k.txt.gz
/bin/wget https://ttic.uchicago.edu/~kgimpel/comsense_resources/dev1.txt.gz
/bin/wget https://ttic.uchicago.edu/~kgimpel/comsense_resources/dev2.txt.gz
/bin/wget https://ttic.uchicago.edu/~kgimpel/comsense_resources/test.txt.gz

/bin/gunzip train100k.txt.gz
/bin/gunzip dev1.txt.gz
/bin/gunzip dev2.txt.gz
/bin/gunzip test.txt.gz

cd ..