wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
/tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vGU6BvCW7xDIw4Nqj5o8x2TIXIcd5KEA' \
-O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vGU6BvCW7xDIw4Nqj5o8x2TIXIcd5KEA" -O model.pth && rm -rf /tmp/cookies.txt

cp model.pth output/TL_CodeSum


wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
/tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1F6rz_AHOW9LrQ-uqt251cUUjZIcKPxSd' \
-O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1F6rz_AHOW9LrQ-uqt251cUUjZIcKPxSd" -O model.pth && rm -rf /tmp/cookies.txt

cp model.pth output/Funcom

