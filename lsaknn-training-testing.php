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

// $ini_k[0] = intval(634);
// $ini_k[1] = intval(643);
// $ini_k[2] = intval(652);
// $ini_k[3] = intval(661);
// $ini_k[4] = intval(670);
// $ini_k[5] = intval(679);
// $ini_k[6] = intval(688);
// $ini_k[7] = intval(697);
// $ini_k[8] = intval(706);
// $ini_k[9] = intval(715);
// $ini_k[0] = intval($svd['K']*5/100);
// $ini_k[1] = intval($svd['K']*10/100);
// $ini_k[2] = intval($svd['K']*15/100);
// $ini_k[3] = intval($svd['K']*20/100);
// $ini_k[4] = intval($svd['K']*25/100);
// $ini_k[5] = intval($svd['K']*30/100);
// $ini_k[6] = intval($svd['K']*35/100);
// $ini_k[7] = intval($svd['K']*40/100);
// $ini_k[8] = intval($svd['K']*45/100);
// $ini_k[9] = intval($svd['K']*50/100);
// $ini_k[10] = intval($svd['K']*55/100);
// $ini_k[11] = intval($svd['K']*60/100);
// $ini_k[12] = intval($svd['K']*65/100);
// $ini_k[13] = intval($svd['K']*70/100);
// $ini_k[14] = intval($svd['K']*75/100);
// $ini_k[15] = intval($svd['K']*80/100);
// $ini_k[16] = intval($svd['K']*85/100);
// $ini_k[17] = intval($svd['K']*90/100);
// $ini_k[18] = intval($svd['K']*95/100);
// $ini_k[19] = intval($svd['K']);
// print_r("<br>");
// print_r($ini_k);

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
				$svd_datauji[$key][$ke] = minmax($val);
			}
		}
	}
	$datalatihred[$key] = new Labeled($svd[$key]['red'], $json_latih['label']);
	$dataujired[$key] = new Labeled($svd_datauji[$key], $labeluji);

//==========================================================================================================================================//

	$report = new MulticlassBreakdown();
	print_r('<pre><h3>PENGUJIAN FOLD ke 0 KNearestNeighbors K = 5 LSA K ='.$ini_k[$key].'</h3>');
	$estimator = new KNearestNeighbors(5);
	
	$estimator->train($datalatihred[$key]);
	$predictions[$key] = $estimator->predict($dataujired[$key]);
	$results[$key] = $report->generate($predictions[$key], $labeluji);
	$akurasi[$key] = $results[$key]['overall']['accuracy'];
	$fmeasure[$key] = $results[$key]['overall']['f1_score'];
	$presisi[$key] = $results[$key]['overall']['precision'];
	$recall[$key] = $results[$key]['overall']['recall'];
	print_r("<br>AKURASI   = " . $results[$key]['overall']['accuracy']);
	print_r("<br>F-MEASURE = " . $results[$key]['overall']['f1_score']);
	print_r("<br>PRECISION = " . $results[$key]['overall']['precision']);
	print_r("<br>RECALL    = " . $results[$key]['overall']['recall']);
	print_r("<br><br>");
	print_r($results[$key]);

//===========================================================================================================================================//
// //SIMPAN MODEL
// //menyimpan model agar tidak dilatih ulang setiap saat.
// $modelManager = new ModelManager();
// $modelManager->saveToFile($classifier, __DIR__.'/lsaknn.model');


}