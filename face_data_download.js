const examplejson = require('./custom_json/113.json');
const fs = require('fs');
const request = require('request');
let image_results = examplejson.images_results;
let txtUrlCounter = 11301;
let imageNumberCounter = 11301;

// Writes data to urls.txt
const WriteToFile = () => {
	for (let i = 0; i < image_results.length; i++) {
		fs.appendFile('validationUrls.txt', `${txtUrlCounter}. ${image_results[i].original}\n`, function (err) {
		if (err) throw err;
		}
	);
	txtUrlCounter++;
	}
};

// Download single image
const DownloadImage = (url, path, callback) => {
	request.head(url, (err, res, body) => {
		if (err) console.log('Error!!');
		else request(url).pipe(fs.createWriteStream(path)).on('close', callback);
	});
};

// Downloads multiple images
const DownloadMultipleImages = () => {
	for (let j = 0; j < image_results.length; j++) {
		let url = image_results[j].original;
		let path = './validationSet/image' + imageNumberCounter + '.png';

		DownloadImage(url, path, () => {
			console.log(`Downloaded url: ${url}`);
		});
		imageNumberCounter++;
	}
};

WriteToFile();
DownloadMultipleImages();
