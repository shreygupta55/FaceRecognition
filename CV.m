close all;
clc

%Read all the images into seperate variables
I1 = imread('C:\shrey college\Semester 1\Computer Vision\Project2\Training Images\subject01.normal.jpg');
I2 = imread('C:\shrey college\Semester 1\Computer Vision\Project2\Training Images\subject02.normal.jpg');
I3 = imread('C:\shrey college\Semester 1\Computer Vision\Project2\Training Images\subject03.normal.jpg');
I4 = imread('C:\shrey college\Semester 1\Computer Vision\Project2\Training Images\subject07.normal.jpg');
I5 = imread('C:\shrey college\Semester 1\Computer Vision\Project2\Training Images\subject10.normal.jpg');
I6 = imread('C:\shrey college\Semester 1\Computer Vision\Project2\Training Images\subject11.normal.jpg');
I7 = imread('C:\shrey college\Semester 1\Computer Vision\Project2\Training Images\subject14.normal.jpg');
I8 = imread('C:\shrey college\Semester 1\Computer Vision\Project2\Training Images\subject15.normal.jpg');

ImageArray = [I1 I2 I3 I4 I5 I6 I7 I8];
%Store each image into seperate vectors which are in 1D format. There must
%be 195 cols *231 rows for each vector but only one column.

V1 = imresize(I1,[45045 1]);
V2 = imresize(I2,[45045 1]);
V3 = imresize(I3,[45045 1]);
V4 = imresize(I4,[45045 1]);
V5 = imresize(I5,[45045 1]);
V6 = imresize(I6,[45045 1]);
V7 = imresize(I7,[45045 1]);
V8 = imresize(I8,[45045 1]);


%Now we calculate the mean value of the vectors . First add all the
%individual vectors and then divide the summation by 8 as there are 8
%training images in the set.
% First we need to convert them by double, as intially it is a uint8
% variable and wont let the value go above 255. Hence, to allow the value
% to go above 255, we perform the double converion of the images.

V1temp = double(V1);
V2temp = double(V2);
V3temp = double(V3);
V4temp = double(V4);
V5temp = double(V5);
V6temp = double(V6);
V7temp = double(V7);
V8temp = double(V8);

sum = V1temp +V2temp+V3temp+V4temp+V5temp+V6temp+V7temp+V8temp;
mean = sum./8;
%figure;
%meanShow = imresize(mean,[231,195]);
%subplot(1,1,1), pcolor(flipud(meanShow)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);


%Now we will construct the normalized vectors of each image. This is
%obtained by substrating the mean calculated value from each of the image
%vectors.

V1normal = V1temp-mean;
V2normal = V2temp-mean;
V3normal = V3temp-mean;
V4normal = V4temp-mean;
V5normal = V5temp-mean;
V6normal = V6temp-mean;
V7normal = V7temp-mean;
V8normal = V8temp-mean;

% we will merge all the norlam vectors into one matrix so that we can get
% one dataset of all the normal vectors of the training images. This is
% represented by the array A.

A = [V1normal V2normal V3normal V4normal V5normal V6normal V7normal V8normal];

%Now we will compute both the covariance matrix. Covariance matrix is
%obtained by the multiplication of a matrix and it's transpose.

%C = A*A.'; 
L = A.'*A;

% Now we will obtain the eigenvectors and eigenvalues from the covariance
% matrix we have obtained. These can be obtained with the help of eigen
% functions. 

[V,D] = eigs(L,8,'largestabs');

%In the above line of code V will denote the matrix of the column vectors
%while D will denote the diagonal eigenvalues of the eigenvectors. The
%largestabs help in getting the 8 largest eigenvectors from L and store
%them in V. The corresponding values are stored in D.

%Now we mention the code to get the eigenvectors of the covariance matrix
%C. Since the computations were very large, we do this process to get the
%eigen vectors of these dimensions.

U = A*V;

%We can display the eigenfaces by the following code.

eigen1 = reshape(U(:,1),231,195);
%figure;
%eigen1Show = imresize(eigen1,[231,195]);
%subplot(1,1,1), pcolor(flipud(eigen1Show)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);

eigen2 = reshape(U(:,2),231,195);
%figure;
%eigen2Show = imresize(eigen2,[231,195]);
%subplot(1,1,1), pcolor(flipud(eigen2Show)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);

eigen3 = reshape(U(:,3),231,195);
%figure;
%eigen3Show = imresize(eigen3,[231,195]);
%subplot(1,1,1), pcolor(flipud(eigen3Show)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);

eigen4 = reshape(U(:,4),231,195);
%figure;
%eigen4Show = imresize(eigen4,[231,195]);
%subplot(1,1,1), pcolor(flipud(eigen4Show)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);

eigen5 = reshape(U(:,5),231,195);
%figure;
%eigen5Show = imresize(eigen5,[231,195]);
%subplot(1,1,1), pcolor(flipud(eigen5Show)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);

eigen6 = reshape(U(:,6),231,195);
%figure;
%eigen6Show = imresize(eigen6,[231,195]);
%subplot(1,1,1), pcolor(flipud(eigen6Show)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);

eigen7 = reshape(U(:,7),231,195);
%figure;
%eigen7Show = imresize(eigen7,[231,195]);
%subplot(1,1,1), pcolor(flipud(eigen7Show)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);

eigen8 = reshape(U(:,8),231,195);
%figure;
%eigen8Show = imresize(eigen8,[231,195]);
%subplot(1,1,1), pcolor(flipud(eigen8Show)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);

%Now we will write the code to get the PCA coefficients of the training
%images.The coefficients are denoted by PCAi

PCA1 = U.'*V1normal;
PCA2 = U.'*V2normal;
PCA3 = U.'*V3normal;
PCA4 = U.'*V4normal;
PCA5 = U.'*V5normal;
PCA6 = U.'*V6normal;
PCA7 = U.'*V7normal;
PCA8 = U.'*V8normal; 

PCAarray = [PCA1 PCA2 PCA3 PCA4 PCA5 PCA6 PCA7 PCA8];
%Since we have our training set ready along with the PCA coeffieicients, we
%will start testing our algorithm against the test images. 


% We will ask the user to select the image from the system
[FileName, Path]  = uigetfile('*.bmp ; *.png ; *.jpg','Select the test image');
p1 = imread(strcat(Path, FileName));

%we resize the image in the form of a 1D column vector
Image = imresize(p1,[45045 1]);
Imagetemp = double(Image);

%we calculate the normal vector by subtracting mean vector from the image
%vector
ImageNormal = Imagetemp - mean;
figure;
ImageNormalShow = imresize(ImageNormal,[231,195]);
subplot(1,1,1), pcolor(flipud(ImageNormalShow)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);

%Projection onto face space is computed as follows
proj = U.'*ImageNormal;

%Reconstruct the image from the eigenface
ImageReconstructed = U*proj;
figure;
ImageReconstructedShow = imresize(ImageReconstructed,[231,195]);
subplot(1,1,1), pcolor(flipud(ImageReconstructedShow)), shading interp, colormap(gray), set(gca, 'Xtick', [], 'Ytick', []);

%The distance between the input test image and it's reconstruction
d0 = norm(ImageReconstructed - ImageNormal);

% construct an array which calculates the distances of the projection of
% the input test image from the projections of eigenfaces computed from the
% training set.
result = zeros(size(8));
for i = 1:8
    q = PCAarray(:,i);
    dist = norm(proj-q);
    result(i)= dist;
end

%Choose the minimum distance value alongwith the index.
[distJ,ind] =min(result.');

%Compare the distance value from the threshold Threshold1. If it is lower
%than that, then it will be recognized as the image in the correspong index
%of Image Array.

Threshold1 = 6e+07;
if distJ<Threshold1
   recognisedImage = ImageArray(ind);
    figure;
    imshow(recognisedImage);
end


