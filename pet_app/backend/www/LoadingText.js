//担当:A

//ローディング中テキスト表示用
function showLoadingText() {
  const loading = document.getElementById("loadingText");
  if(loading){
    loading.style.display = "block";
  }
}

//ローディング中テキスト非表示用
function hideLoadingText() {
 const loading = document.getElementById("loadingText");
  if(loading){
    loading.style.display = "none";
  }
}
