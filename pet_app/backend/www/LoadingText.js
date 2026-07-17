//担当:A

//ローディング中テキスト表示用
function showLoadingText() {
  const loading = document.getElementById("loadingText");
  if(loading){
    loading.style.display = "block";
  }
}

function hideLoadingText() {
 const loading = document.getElementById("loadingText");
  if(loading){
    loading.style.display = "none";
  }
}
