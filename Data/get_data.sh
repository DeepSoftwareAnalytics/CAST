wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
/tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jwSYdODAaQgLtEdwYJkMEGCN3gtv-nZ8' \
-O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1jwSYdODAaQgLtEdwYJkMEGCN3gtv-nZ8" -O TL_CodeSum.tar.bz2 && rm -rf /tmp/cookies.txt

tar -jxvf TL_CodeSum.tar.bz2


wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
/tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Cx3TYtVg2ZeGPHSBQAEMOBtsoAIpMZot' \
-O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Cx3TYtVg2ZeGPHSBQAEMOBtsoAIpMZot" -O Funcom.tar.bz2 && rm -rf /tmp/cookies.txt

tar -jxvf Funcom.tar.bz2
