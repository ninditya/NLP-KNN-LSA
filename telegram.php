<?php

// error_reporting(0);

// https://api.telegram.org/bot1299833329:AAHPzf6fAuvFznSxh6NOtstsETnONk0m_rY/setWebhook?url=https://9594bbeda602.ngrok.io/SKRIPSI/Chatbot-VSM/telegram/index.php

require './vendor/autoload.php';

// $chat_id = 123160175;

$token = "bot"."1299833329:AAHPzf6fAuvFznSxh6NOtstsETnONk0m_rY";
$proxy = "";

include "koneksi_telegram.php";

$updates = file_get_contents("php://input");

$updates = json_decode($updates,true);
$pesan_awal = $updates['message']['text'];
$dokumen_id = $updates['message']['document']['file_id'];
$pesan = $updates['message']['text'];
$chat_id = $updates['message']['chat']['id'];

// //DISINI ROUTING PERTANYAAN
// $sql = mysqli_query($conn, "SELECT * FROM data_id WHERE id_chat='$chat_id'");
// $d = mysqli_fetch_array($sql);

// include 'lsaknn-telegram_ir.php';
// include 'knn-telegram_ir.php';
// include 'lsasvm-telegram_ir.php';
include 'svm-telegram_ir.php';

// simpan query jawaban disini
mysqli_query($conn, "INSERT INTO riwayat_query(id_history, id_chat, pertanyaan, jawaban, waktu) VALUES ('','$chat_id','$pesan_awal','$id_jawaban',now())");
// $id_media = 'AgACAgUAAxkBAAICCl-qNBuUZXdYeKFJHbTpsbgTK4r7AAKCqjEbXjJRVWEWluuvLZEycUgkbXQAAwEAAwIAA3kAAwQfAAIeBA';
$url = "https://api.telegram.org/$token/sendMessage?parse_mode=markdown&chat_id=$chat_id&text=$pesan_balik";
// $url = "https://api.telegram.org/$token/sendPhoto?parse_mode=markdown&chat_id=$chat_id&text=$pesan_balik&photo=$id_media";

$ch = curl_init();

if($proxy==""){
	$optArray = array(
		CURLOPT_URL => $url,
		CURLOPT_RETURNTRANSFER => true,
		CURLOPT_CAINFO => "C:\cacert.pem"	
	);
}
else{ 
	$optArray = array(
		CURLOPT_URL => $url,
		CURLOPT_RETURNTRANSFER => true,
		CURLOPT_PROXY => "$proxy",
		CURLOPT_CAINFO => "C:\cacert.pem"	
	);	
}

curl_setopt_array($ch, $optArray);
$result = curl_exec($ch);

$err = curl_error($ch);
curl_close($ch);	

if($err<>"") echo "Error: $err";
else echo "<br>Pesan Terkirim";

?>