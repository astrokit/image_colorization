
fnames=()
fnames[0]=01_gradientR.png
fnames[1]=02_binaryR.png
fnames[2]=03_edgesB.png
fnames[3]=04_edgesG.png
fnames[4]=05_edgesR.png
fnames[5]=06_translatedR.png
fnames[6]=07_alignedB.png
fnames[7]=08_alignedG.png
fnames[8]=09_combined.png
fnames[9]=10_combined_cropped.png
fnames[10]=11_cartoon.png

for tc in "boats" "cathedral" "chalice" "city" "emir" "generators" "three_girls" "train"
do
    if [ -d data/ref_x64/${tc} ]
    then
      mkdir -p dif/${tc}
      for fname in ${fnames[*]}
      do
        if [ -f output/${tc}/${fname} ]
        then
          ref_size="$(identify -ping -format '%wx%h' data/ref_x64/${tc}/${fname})"
          output_size="$(identify -ping -format '%wx%h' output/${tc}/${fname})"
          convert data/ref_x64/${tc}/${fname} output/${tc}/${fname} -compose difference -composite -negate -contrast-stretch 0 dif/${tc}/${fname}
          if [[ $ref_size != $output_size ]]
          then
            convert -background 'rgba(0, 0, 0, .75)' -font DejaVu-Sans -fill white -pointsize 15 \
              label:" Different Sizes " -splice 0x3 dif/${tc}/${fname} +swap -gravity north \
              -geometry +0+10 -composite dif/${tc}/${fname}
          fi
        else
          convert -size 300x300 xc:red dif/${tc}/${fname}
        fi
      done
    fi
done


