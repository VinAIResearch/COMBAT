wget -P ./data http://cs231n.stanford.edu/tiny-imagenet-200.zip;
mkdir ./data/TinyImageNet;
mkdir ./data/temps;
unzip ./data/tiny-imagenet-200.zip -d ./data/temps;
mv ./data/temps/tiny-imagenet-200/* ./data/TinyImageNet;
rm -r ./data/temps;
rm ./data/*.zip;
echo "Download Completed";
