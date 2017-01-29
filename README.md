# CS510 Assignments (Spring 17)
## Instructions for working with branches
1. First off, try to make commits as small and incremental as possible, so that there is not a huge chunk of code that changes with a single commit.
2. No one pushes to `master` or `development`
3. Everybody works by basing their work off `development` branch, which in turn is based off `mater`. For example, suppose you needed to add gui to the project. You can create a branch `add-gui` derived from the `development` branch to implement GUI.
 ```
 git checkout -b myfeature develop
```
4. When you are done with your feature, merge back without fast forwarding like (avoid fast forwarding as it results in loss of history)
 ```
git checkout development
git merge --no-ff add-gui
```
5. You can now safely delete your own feature branch, and push the changes to remote `development` branch
 ```
git branch -d add-gui
git push origin development
```    
5. Eventually, we merge back `development` brach back into the `master` branch
