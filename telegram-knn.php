<?php
require './vendor/autoload.php';
// $pesan = 'DAFTAR#Ninditya Salma Nur Aini#123160175';
// $chat_id = 123160175;

ini_set('memory_limit', '-1');

use Rubix\ML\ModelManager;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\Persisters\Filesystem;

include './fungsi.php';
include './fungsi_svd.php';
include "koneksi_telegram.php";

$stemmerFactory = new \Sastrawi\Stemmer\StemmerFactory(); //Panggil Kelas Stemmer
$stemmer  = $stemmerFactory->createStemmer();

//TABLE TF

//TABLE TFIDF

//TABLE U

//TABLE S

//TABLE V


//DISINI ROUTING PERTANYAAN
$sql = mysqli_query($conn, "SELECT * FROM data_id WHERE id_chat='$chat_id'");
$d = mysqli_fetch_array($sql);

if (strpos($pesan,"AFTAR#")>0) {
	$datas = explode("#",$pesan);
	// print_r($datas);
	$nama = $datas[1];
	$nim = $datas[2];
	$sql = "UPDATE data_id SET nama='$nama',NIM='$nim' WHERE id_chat='$chat_id'";
	if(mysqli_query($conn,$sql)) {
		$pesan_balik = "Terima kasih Data Anda sudah kami simpan. Ada yang bisa saya bantu?";
	} else {
		$pesan_balik = "Data gagal disimpan silahkan coba lagi. Pastiakan format benar seperti berikut DAFTAR%23NAMA%23NIM";
	}
} elseif ($d['NIM'] == 0) {
	$pesan_balik = 'Format Anda salah. Mohon input nama dan nim dengan format berikut DAFTAR%23NAMA%23NIM';
	if (mysqli_num_rows($sql) == 0) {
		$pesan_balik = 'Selamat datang di Layanan Informasi Akademik Jurusan Infromatika UPN "Veteran" Yogyakarta. Untuk langkah awal menggunakan layanan ini, input nama dan nim dengan Format berikut DAFTAR%23NAMA%23NIM';
		$sql = mysqli_query($conn, "INSERT INTO data_id (id, id_chat) VALUES ('','$chat_id')");
		// $sql = mysqli_query($conn,"insert into data_id values ('', $chat_id','','')");
	}
} elseif (strpos($pesan,"TRANSKRIPNILAI#")>0) {
	$datas = explode("#",$pesan);
	$nama = $datas[1];
	$nim = $datas[2];
	$sql = "INSERT INTO transkrip (id, id_chat, nama, nim) VALUES ('','$chat_id','$nama','$nim')";
} else { 
	if(!empty($pesan))
	{
		//DISINI PREDICT KNN DAN RETRIEVE ANSWER
		//TEKS PREPROCESSING DATA BARU
		print_r("</pre><hr><pre><h1>HASIL PREPROCESSING DATA BARU</h1><br>");

		print_r("<h3>HASIL STEMMING DATA BARU </h3><br>");

		print_r("</pre>");

		echo "<hr><pre><h1>LATENT SEMANTIC ANALYSIS DATA BARU</h1>";

		echo "<hr><pre><h1>PREDIKSI K Nearest Neighbors</h1>";
		//memunculkan model latih yang disimpan
		$modelManager = new ModelManager();
		$model = $modelManager->restoreFromFile(__DIR__.'/model/TA-knn.model');
		print_r("<h3>PAKAI MODEL MANAGER</h3><br>");
		// print_r($model);

		print_r("<h3>LABEL DATA LATIH</h3><br>");
		$databaru = new Unlabeled($matrix_q_red);

		$predictions = $model->predict($databaru);

		print_r("<h3>LABEL DATA BARU</h3><br>");
		$hasilpredict=$predictions[0];

		//AMBIL DATA LATIH
		$i=0;
		print_r("<h3>HASIL TEKS PREPROCESSING KANDIDAT JAWABAN </h3><br>");


		echo "<hr><pre><h1>LATENT SEMANTIC ANALYSIS KANDIDAT JAWABAN</h1>";

		print_r("<h3>Matrix reduksi kandidat jawaban</h3><br>");


		echo "<hr><pre><h1> COSINE SIMILARITY DATA BARU DAN KANDIDAT JAWABAN</h1>";


		echo "<hr><pre><h1> EKSTRAKSI JAWABAN </h1>";

		$pesan_balik = $gabung_baru[0]['jawaban'] ;
		echo $pesan_balik ;
	}else {
		$pesan_balik = 'bentar yaa. kodingannya nyasar '. $Query;
	}
}

?>