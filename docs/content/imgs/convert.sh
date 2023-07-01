for f in *; do
    if [ -d "$f" ]; then
        cd $f
        for g in *.pdf; do     
            pdftoppm ./"$g" ./"${g%.pdf}" -png
        done
        cd ..
    fi
done
