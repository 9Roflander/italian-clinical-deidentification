#!/bin/zsh
# Builds the archive the Colab notebooks unpack: the committed code as a git bundle, gold_v2 and the synthetic examples,
# with a manifest of hashes. The archive holds private data: keep it on your own Drive, never share a link to it.
#
#   cd ../repo && ../colab/build_transfer.sh nj148_transfer_e3x_bert.tar.gz
#
# The notebook checks that the bundle's commit, the gold hash and every synthetic file match the manifest.
set -e
NAME=${1:?archive name, e.g. nj148_transfer_e3x_bert.tar.gz}
OUT=${2:-$PWD/..}
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "commit your changes first"; exit 1; }
B=$(mktemp -d); chmod 700 $B
git bundle create $B/repo.bundle research-reconstruction 2>/dev/null
cp data_secure/gold/gold_v2.jsonl $B/
for f in synthetic_data/fold_*/*/examples.jsonl; do mkdir -p $B/$(dirname $f); cp $f $B/$f; done
python3 - $B <<'EOF'
import hashlib, json, pathlib, subprocess, sys
b = pathlib.Path(sys.argv[1]); h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
files = {str(p.relative_to(b)): h(p) for p in sorted(b.glob("synthetic_data/fold_*/*/examples.jsonl"))}
m = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "branch": "research-reconstruction",
     "gold_sha256": h(b / "gold_v2.jsonl"), "files": files}
(b / "transfer_manifest.json").write_text(json.dumps(m, indent=1))
print("commit", m["commit"][:7], "| synthetic files", len(files))
EOF
xattr -rc $B
(cd $B && COPYFILE_DISABLE=1 tar --no-xattrs -czf $OUT/$NAME repo.bundle gold_v2.jsonl transfer_manifest.json synthetic_data)
rm -rf $B; chmod 600 $OUT/$NAME
echo "wrote $OUT/$NAME"
