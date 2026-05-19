find ./data/re10k -type d -name images | while read imgdir; do
  count=$(find "$imgdir" -maxdepth 1 -type f | wc -l)
  if [ "$count" -gt 10 ]; then
    basename "$(dirname "$imgdir")"
  fi
done > folders_with_many_images.txt