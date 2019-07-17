# CS510 Assignments (Spring 17)

<p align="center">
  <img src="https://gurumulay.github.io/images/vision/image-computation/img-comp1.png?raw=true" alt=""/>
</p>

<p align="center">
  <img src="https://gurumulay.github.io/images/vision/image-computation/clustering.png?raw=true" alt=""/>
</p>

<p align="center">
  <img src="https://gurumulay.github.io/images/vision/image-computation/saliency.png?raw=true" alt=""/>
</p>


<video width="730" autoplay loop="loop" controls="false">
  <source src="https://youtu.be/Ajp-H0SiAGo" type="video/mp4">
  <p>Unfortunately, your browser doesn't support HTML5 embedded videos. Here is
    a <a href="https://youtu.be/Ajp-H0SiAG">link to the video</a> instead.</p>
</video>


## Instructions for working with branches
1. First off, try to make commits as small and incremental as possible, so that there is not a huge chunk of code that changes with a single commit.
2. No one pushes to `master` or `development`
3. Everybody works by basing their work off `development` branch, which in turn is based off `master`. For example, suppose you needed to add gui to the project. You can create a branch `add-gui` derived from the `development` branch to implement GUI.

 ```
 git checkout -b add-gui develop
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

5. Eventually, we merge back `development` branch back into the `master` branch
