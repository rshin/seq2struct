#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Please specify directory containing Spider files."
    exit 1
fi

BASE=$(realpath $(dirname $0))

# Verify checksums
pushd $1 >& /dev/null
sha256sum -c ${BASE}/SHA256SUMS || exit 1
popd >& /dev/null

# Re-generate 'sql' to fix bad parsing
SPIDER_BASE=$(realpath ${BASE}/../../third_party/spider)
cp $1/tables.json ${BASE}
for input in train_others train_spider dev; do
    echo Procesing $input
    cp $1/${input}.json ${BASE}
    if [[ -e ${BASE}/${input}.json.patch ]]; then
        pushd ${BASE} >& /dev/null
        patch < ${input}.json.patch
        popd >& /dev/null
    fi
        python ${SPIDER_BASE}/preprocess/parse_raw_json.py \
        --tables ${BASE}/tables.json \
        --input ${BASE}/${input}.json \
        --output ${BASE}/${input}.json
    echo
done

# Create augmented data
SYNTAXSQL_BASE=$(realpath ${BASE}/../../third_party/syntaxSQL)
echo "Generating augmented data..."
python ${SYNTAXSQL_BASE}/generate_wikisql_augment.py --output ${BASE}/train_wikisql_augment.json
cp ${SYNTAXSQL_BASE}/data_augment/wikisql_tables.json ${BASE}/tables_wikisql_augment.json