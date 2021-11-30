<?php 
$mysql_host = "localhost";
$mysql_user = "root";
$mysql_pass = "";
$mysql_dbname = "ifva";

// ==== END / variabel must be adjusted ====


$conn = mysqli_connect($mysql_host, $mysql_user, $mysql_pass);
if(! $conn ) {
  die('Could not connect: ' . mysqli_error());
}

$db_selected = mysqli_select_db($conn,$mysql_dbname);
if (!$db_selected) {
  die ('Can\'t use foo : ' . mysqli_error() .'<br>');
}

 ?>