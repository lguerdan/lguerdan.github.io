rm -rf ../build
mkdir ../build
cp -r _site/* ../build/
branch=$(git symbolic-ref HEAD | sed -e 's,.*/\(.*\),\1,')
git checkout site-build
cp -r ../build/* .
git add .
Git commit -m “deployment”
git push origin site-build
git checkout $branch
