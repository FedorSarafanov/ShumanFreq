#!/bin/bash
for i in $( ls | grep x); do
    echo item: $i
    sed 's/ //g' $i > _$i
    # sed -e '$d' $i > $i
done