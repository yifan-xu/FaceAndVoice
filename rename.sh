images=$(for i in `ls *.jpg`; do LEN=`expr length $i`; echo  $i; done | sort -n)
j=1
for i in $images; do
  new=$(printf "%03d.png" "$j") #04 pad to length of 4
  mv -i -- "$i" "$new"
  let j=j+1
   #we can add a condition on j to rename just the first 999 images.
done
