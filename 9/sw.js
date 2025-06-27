self.addEventListener('install', event => {
    console.log('Service Worker 安装完成');
  });
  
  self.addEventListener('fetch', event => {
    // 不做任何拦截，PWA最低要求
  });
  