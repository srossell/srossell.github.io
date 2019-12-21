Title: Pelican github pages workflow
Date: 2019-12-21
Summary: Building a website using pelican, and then deploying it in github
pages.
Image: /images/blog/tech/blog_191221_pelican/pelican_icon.png
Tags: github
Slug: pelican-github-workflow

## Building an deploying a website with Pelican and github.

This is short description of how I deployed this website using pelican and
githubpages.

The theme of this page is a modification of Claudio Walser's [fh5co
theme](https://github.com/claudio-walser/pelican-fh5co-marble). The workflow is
modified from a workflow by Joel Zeldes at
[anotherdatum](http://anotherdatum.com/pelican-and-github-pages-workflow.html).

First create a repository with a `master` and a `source` branch. In teh
`source` branch is where you will develop your pelican website. The `master`
branch is used for publishing the site.

You'll need to create a folder in `.git/hooks/pre-push` with the follwing
content.

```bash
#!/bin/sh
while read local_ref local_sha remote_ref remote_sha
do
        if [ "$remote_ref" = "refs/heads/source" ]
        then
                echo 'pushing output folder (production version) to master...'
                pelican content -o output -s publishconf.py
                ghp-import --branch=master output
                git push --force git@github.com:srossell/srossell.github.io.git master
                pelican content -o output
        fi
done

exit 0
```

To get the workflow to work in my case, I added the `--branch=master` option
for `ghp-import` and pushed to `master`. Also, I had to create an ssh key. I
also added the `--force` option to git push.


