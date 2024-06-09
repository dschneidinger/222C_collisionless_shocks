ssh dschneidinger@summit.olcf.ornl.gov
\
while getopts 'p:' flag; do
  case "${flag}" in
    p) ${OPTARG} ;;
    v) verbose = 'true' ;;
  esac
done

