<?php
declare(strict_types=1);
namespace IFVABOTKNN;

require './vendor/autoload.php';
// ini_set('memory_limit', '1024M'); // or you could use 1G
ini_set('memory_limit', '-1');

use Rubix\ML\ModelManager;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Dataset;

include './fungsi.php';
include './fungsi_svd.php';

$connect = connectDB('ifva');

$json_datatraining = file_get_contents("json/datatraining.json");
$data_training = json_decode($json_datatraining, TRUE);
$red = $data_training['red'][0];
print_r("<pre>");
// print_r($red);

$json_uji = file_get_contents("json/datauji.json");
$data_uji = json_decode($json_uji, TRUE);
// print_r($data_uji);
foreach ($data_uji as $key => $value) {
	$tf[$key][0] = $data_uji[$key]['tf'];
	$tf_red[$key] = perkalian_matrix($tf[$key], $red);
	foreach ($tf_red as $ke => $value) {
		foreach ($value as $kunci => $val) {
			$tf_red_normal[$ke] = minmax($val);
		}
	}
	$labeluji[$key] = $data_uji[$key]['label'];
}
// print_r($labeluji);
$dataujired = new Labeled($tf_red_normal, $labeluji);

$modelManager = new ModelManager(); //memunculkan model latih yang disimpan
echo "<hr><pre><h1>PREDIKSI K Nearest Neighbors</h1>";
$model = $modelManager->restoreFromFile(__DIR__.'/lsaknn.model');
$predictions = $model->predict($dataujired);

print_r($predictions);
