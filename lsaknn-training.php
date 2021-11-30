<?php

declare(strict_types=1);
namespace IFVABOTKNN;

require './vendor/autoload.php';
// ini_set('memory_limit', '1024M'); // or you could use 1G
ini_set('memory_limit', '-1');

use Rubix\ML\GridSearch;
use Rubix\ML\ModelManager;
use Rubix\ML\Backends\Tasks\TrainAndValidate;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\Classifiers\SVC;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Kernels\SVM\Linear;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Tfidf;
use svmmodel;
use svm;

include './fungsi.php';
include './fungsi_svd.php';

$connect = connectDB('ifva');

$data_uji = file_get_contents("json/datauji.json");
$json_uji = json_decode($data_uji, TRUE);
foreach ($json_uji as $key => $value) {
	$labeluji[$key] = $value['label'];
}
$data_latih = file_get_contents("json/datalatih.json");
$json_latih = json_decode($data_latih, TRUE);

$json_svd = file_get_contents("json/svd.json");
$svd = json_decode($json_svd, TRUE);

// print_r("<hr><h1>SVD datalatih</h1><br>");
$d_datalatih = count($json_latih['tfidf']);
$t_datalatih = count(matrixTranspose($json_latih['tfidf']));
print_r("<pre>");
print_r("TERM DATA LATIH = ".$t_datalatih);
print_r("<br>");
print_r("DOKUMEN DATA LATIH = ".$d_datalatih);

$ini_k[0] = intval(661);

foreach ($ini_k as $key => $value) {
	$svd_red[$key] = array();
	$svd_red[$key]['sp'] = matrixConstruct($svd['S'], $value, $value);
	$svd_red[$key]['up'] = matrixConstruct($svd['U'], count($svd['U']), $value);
	$svd_red[$key]['vtp'] = matrixConstruct(matrixTranspose($svd['V']), $value, count($svd['V']));
	$newArr[$key] = Array();
	foreach ($svd_red[$key]['sp']  as $k => $val) {
		$r = Array();
		foreach ($val as $v) {
			array_push($r, floatval($v) == 0.0 ? 0 : floatval($v));
		}
		array_push($newArr[$key], $r);
	}
	$spinvers_datalatih[$key] = invert($newArr[$key]);
	// echo "<hr><pre><h1>LATENT SEMANTIC ANALYSIS LATIH</h1>";
	if ($t_datalatih > $d_datalatih) {
		$red[$key] = perkalian_matrix($svd_red[$key]['up'], $spinvers_datalatih[$key]);
	} elseif ( $d_datalatih >= $t_datalatih) {
		$red[$key] = perkalian_matrix(matrixTranspose($svd_red[$key]['vtp']), $spinvers_datalatih[$key]);
	}
	// REDUKSI PADA DATA LATIH
	for ($x=0; $x < $d_datalatih ; $x++) {
		$tf_datalatih_red[$key][$x] = perkalian_matrix(array($json_latih['tfidf'][$x]), $red[$key]);
		foreach ($tf_datalatih_red[$key] as $k => $value) {
			foreach ($value as $kunci => $val) {
				$svd[$key]['red'][$k] = minmax($val);
			}
		}
	}
	//REDUKSI PADA DATA UJI
	$x=0;
	foreach ($json_uji as $k => $value) {
		$tf_uji_red[$key][$k] = perkalian_matrix(array($json_uji[$k]['tf']),$red[$key]);
		foreach ($tf_uji_red[$key] as $ke => $value) {
			foreach ($value as $kunci => $val) {
				$svd_datauji[$key][$ke] = $val;
			}
		}
	}
	$datalatihred[$key] = new Labeled($svd[$key]['red'], $json_latih['label']);
	$dataujired[$key] = new Labeled($svd_datauji[$key], $labeluji);

//==========================================================================================================================================//

	$report = new MulticlassBreakdown();
	print_r('<pre><h3>TRAINING DENGAN KNN K=5; LSA K ='.$ini_k[$key].';</h3>');
	$estimator = new KNearestNeighbors(5);
	$estimator->train($datalatihred[$key]);
	//SIMPAN MODEL
	//menyimpan model agar tidak dilatih ulang setiap saat.
	$modelManager = new ModelManager();
	$modelManager->saveToFile($estimator, __DIR__.'/lsaknn.model');
}

$array_training = array('svd_red' =>$svd_red, 'red' =>$red, );
$json_datatraining = json_encode($array_training);
//write json to file
if (file_put_contents("json/datatraining.json", $json_datatraining))
	echo "JSON DATA training file created successfully...<br>";
else 
	echo "Oops! Error creating json file...";

//==============================================================================================================================================//
// $predictions[$key] = $estimator->predict($dataujired[$key]);
// $cek = array_flip(array_values(array_unique($json_latih['label'])));
// //print_r($predictions);
// foreach ($predictions[0] as $kunci => $value) {
// //AMBIL DATA LATIH
// // print_r("<hr><h1>KANDIDAT JAWABAN </h1>");
// 	$query_ptyn[$kunci] = mysqli_query($connect, "SELECT * FROM datalatih WHERE label='$value'");
// 	$kandidat[$kunci]= array();
// 	while ($d = mysqli_fetch_array($query_ptyn[$kunci])) {
// 		$kandidat[$kunci][$d['id']-1]['jawaban'] = $d['jawaban'];
// 	}
// 	$alamat[$kunci] = $cek[$value];
// 	$json_kandidat = file_get_contents("json/datakandidat.json");
// 	$data_kandidat = json_decode($json_kandidat, TRUE);
// 	$tfidfkandidat[$kunci] = $data_kandidat['tfidf']['tfidf'][$alamat[$kunci]];
// 	$termkandidat[$kunci] = $data_kandidat['term'][$alamat[$kunci]];

// 	foreach($array_uji as $k => $val)	{
// 		$tf_q_kandidat[0] = query($array_uji[$k], $termkandidat);
// 		//echo "<hr><pre><h1> COSINE SIMILARITY DATA BARU DAN KANDIDAT JAWABAN</h1>";
// 		foreach ($tfidfkandidat[$kunci] as $key => $val) {
// 			$nilaicossim[$kunci][$key][$k]= cosinesimilarity($tf_q_kandidat[$k],matrixTranspose(array($val)));
// 			foreach ($nilaicossim[$kunci][$key][$k] as $kunci => $v) {
// 				$sorted[$kunci][$key] = $v[0];
// 				arsort($sorted[$kunci], SORT_NUMERIC);
// 			}
// 		}
// 	}
// 	$coba = 0;
// 	foreach ($sorted[$kunci] as $key => $value) {
// 		$gabung_idx[$kunci][$coba]['jawaban']=$kandidat[$kunci][$key]['jawaban'];
// 		$coba++;
// 	}
// 	$jawaban[$kunci] = $gabung_idx[$kunci][0]['jawaban'];
// }
// print_r($jawaban);
// $results= $report->generate($jawaban, $jawaban_asli);
