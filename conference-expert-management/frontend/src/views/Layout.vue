<template>
  <el-container class="layout-container">
    <el-header class="header">
      <div class="logo">
        <el-icon size="24"><User /></el-icon>
        <span>会务专家管理系统</span>
      </div>
      <el-menu
        mode="horizontal"
        background-color="#409EFF"
        text-color="#fff"
        active-text-color="#ffd04b"
        :router="true"
        :default-active="$route.path"
      >
        <el-menu-item index="/">专家列表</el-menu-item>
        <el-menu-item index="/statistics">数据统计</el-menu-item>
      </el-menu>
    </el-header>
    <el-main class="main-content">
      <router-view />
    </el-main>
  </el-container>
</template>

<script>
export default {
  name: 'Layout'
}
</script>

<style scoped>
.layout-container {
  min-height: 100vh;
}

.header {
  background-color: #409EFF;
  color: white;
  display: flex;
  align-items: center;
  padding: 0 20px;
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 20px;
  font-weight: bold;
}

:deep(.el-menu) {
  border-bottom: none;
}

.main-content {
  background-color: #f5f7fa;
  padding: 20px;
}
</style>
