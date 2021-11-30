<?php
declare(strict_types=1);
namespace IFVABOTKNN;

require './vendor/autoload.php';
// ini_set('memory_limit', '1024M'); // or you could use 1G
ini_set('memory_limit', '-1');

use Rubix\ML\ModelManager;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Persisters\Filesystem;


include './fungsi.php';

$connect = connectDB('ifva');

$data_latih = file_get_contents("json/datalatih.json");
$json_latih = json_decode($data_latih, TRUE);
$labellatih = $json_latih['label'];
$tfidflatih = $json_latih['tfidf'];

//TABLE TF UJI
$data_uji = file_get_contents("json/datauji.json");
$json_uji = json_decode($data_uji, TRUE);
foreach ($json_uji as $key => $value) {
	$labeluji[$key] = $value['label'];
	$tfuji[$key] = $value['tf'];
}

$datalatih_tfidf = new Labeled($tfidflatih, $labellatih);
$datauji_tf = new Labeled($tfuji, $labeluji);

//TRAINNIG
print_r("<hr><h1>TRAINING KNN K-5</h1><br>");
$classifier = new KNearestNeighbors(5);
$classifier->train($datalatih_tfidf); // Trainning Hasil TFIDF 

//TESTING
print_r("<hr><h1>TESTING KNN K-5</h1><br>");
$predictions = $classifier->predict($datauji_tf);
$results = $report->generate($predictions,  $labeluji);
echo $results;

//SIMPAN MODEL
//menyimpan model agar tidak dilatih ulang setiap saat.
$modelManager = new ModelManager();
$modelManager->saveToFile($classifier, __DIR__.'/knn.model');


?>