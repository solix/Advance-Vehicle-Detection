import pickle
color_space = 'YCrCb'
orient = 9
pixel_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (64,64)
hist_bins = 64
spatial_feat = True
hist_feat = True
hog_feat = True

# Save parameters to pickle file
dist_pickle = {}
dist_pickle["orient"] = orient
dist_pickle["pixel_per_cell"] = pixel_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["spatial_size"] = spatial_size
dist_pickle["hist_bins"] = hist_bins




t = time.time()
n_samples = 1000
randm_idx = np.random.randint(0,len(cars),n_samples)

test_cars = cars#np.array(cars)#[randm_idx]
test_notcars = not_cars#np.array(not_cars)#[randm_idx]


car_features = extract_features(test_cars, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=9, 
                        pix_per_cell=pixel_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

not_car_features = extract_features(test_notcars, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=9, 
                        pix_per_cell=pixel_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
print(str(time.time() - t) + 'seconds took to compute features...')


X = np.vstack((car_features, not_car_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
dist_pickle["scaler"] = X_scaler


scaled_X = X_scaler.transform(X)

y = np.hstack((np.ones(len(car_features)),np.zeros(len(not_car_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using spatial binning of:',spatial_size,
    'and', hist_bins,'histogram bins')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)

dist_pickle["svc"] = svc

t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test))
print('For these',n_predict, 'labels: ', y_test)
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


filename = "dist.pkl"
pickle.dump( dist_pickle, open( filename, "wb" ) )