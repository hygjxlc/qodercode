<template>
  <div class="expert-list">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>专家列表</span>
          <el-button type="primary" @click="showAddDialog">
            <el-icon><Plus /></el-icon>添加专家
          </el-button>
        </div>
      </template>

      <!-- 搜索栏 -->
      <el-row :gutter="20" class="search-bar">
        <el-col :span="6">
          <el-input
            v-model="searchKeyword"
            placeholder="搜索姓名或专业领域"
            clearable
            @keyup.enter="handleSearch"
          >
            <template #append>
              <el-button @click="handleSearch">
                <el-icon><Search /></el-icon>
              </el-button>
            </template>
          </el-input>
        </el-col>
        <el-col :span="4">
          <el-select v-model="filterField" placeholder="专业领域" clearable @change="handleFilter">
            <el-option label="人工智能" value="人工智能" />
            <el-option label="大数据" value="大数据" />
            <el-option label="云计算" value="云计算" />
            <el-option label="物联网" value="物联网" />
          </el-select>
        </el-col>
        <el-col :span="4">
          <el-select v-model="filterStatus" placeholder="联系状态" clearable @change="handleFilter">
            <el-option label="待联系" value="待联系" />
            <el-option label="已联系" value="已联系" />
            <el-option label="确认参加" value="确认参加" />
            <el-option label="暂不参与" value="暂不参与" />
          </el-select>
        </el-col>
        <el-col :span="4">
          <el-button @click="resetFilter">重置筛选</el-button>
        </el-col>
        <el-col :span="6" style="text-align: right">
          <el-button type="success" @click="exportExperts">
            <el-icon><Download /></el-icon>导出
          </el-button>
          <el-button type="warning" @click="showImportDialog">
            <el-icon><Upload /></el-icon>导入
          </el-button>
        </el-col>
      </el-row>

      <!-- 专家表格 -->
      <el-table :data="experts" v-loading="loading" stripe>
        <el-table-column prop="name" label="姓名" width="120" />
        <el-table-column prop="email" label="邮箱" width="180" />
        <el-table-column prop="phone" label="电话" width="140" />
        <el-table-column prop="professionalField" label="专业领域" width="150" />
        <el-table-column prop="title" label="职称" width="120" />
        <el-table-column prop="contactStatus" label="联系状态" width="100">
          <template #default="scope">
            <el-tag :type="getStatusType(scope.row.contactStatus)">
              {{ scope.row.contactStatus }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200" fixed="right">
          <template #default="scope">
            <el-button size="small" @click="viewDetail(scope.row)">查看</el-button>
            <el-button size="small" type="primary" @click="editExpert(scope.row)">编辑</el-button>
            <el-button size="small" type="danger" @click="deleteExpert(scope.row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <el-pagination
        class="pagination"
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :total="total"
        layout="total, sizes, prev, pager, next"
        @size-change="handleSizeChange"
        @current-change="handleCurrentChange"
      />
    </el-card>

    <!-- 添加/编辑对话框 -->
    <el-dialog
      v-model="dialogVisible"
      :title="isEdit ? '编辑专家' : '添加专家'"
      width="600px"
    >
      <el-form :model="form" :rules="rules" ref="formRef" label-width="100px">
        <el-form-item label="姓名" prop="name">
          <el-input v-model="form.name" />
        </el-form-item>
        <el-form-item label="邮箱" prop="email">
          <el-input v-model="form.email" />
        </el-form-item>
        <el-form-item label="电话" prop="phone">
          <el-input v-model="form.phone" />
        </el-form-item>
        <el-form-item label="专业领域" prop="professionalField">
          <el-input v-model="form.professionalField" />
        </el-form-item>
        <el-form-item label="职称" prop="title">
          <el-input v-model="form.title" />
        </el-form-item>
        <el-form-item label="所属机构" prop="organization">
          <el-input v-model="form.organization" />
        </el-form-item>
        <el-form-item label="联系状态" prop="contactStatus">
          <el-select v-model="form.contactStatus" style="width: 100%">
            <el-option label="待联系" value="待联系" />
            <el-option label="已联系" value="已联系" />
            <el-option label="确认参加" value="确认参加" />
            <el-option label="暂不参与" value="暂不参与" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="submitForm">确定</el-button>
      </template>
    </el-dialog>

    <!-- 导入对话框 -->
    <el-dialog v-model="importDialogVisible" title="导入专家" width="500px">
      <div style="text-align: center; padding: 20px">
        <p style="margin-bottom: 20px">请先下载导入模板，按模板格式填写数据后上传</p>
        <el-button type="primary" @click="downloadTemplate">下载导入模板</el-button>
        <el-divider>或</el-divider>
        <el-upload
          drag
          action="#"
          :auto-upload="false"
          :on-change="handleImport"
          accept=".xlsx,.xls"
        >
          <el-icon class="el-icon--upload"><upload-filled /></el-icon>
          <div class="el-upload__text">
            拖拽文件到此处或 <em>点击上传</em>
          </div>
          <template #tip>
            <div class="el-upload__tip">
              请上传 Excel 文件 (.xlsx, .xls)
            </div>
          </template>
        </el-upload>
      </div>
    </el-dialog>
  </div>
</template>

<script>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { expertApi, excelApi } from '../api/expert'

export default {
  name: 'ExpertList',
  setup() {
    const router = useRouter()
    const loading = ref(false)
    const experts = ref([])
    const currentPage = ref(1)
    const pageSize = ref(10)
    const total = ref(0)
    const searchKeyword = ref('')
    const filterField = ref('')
    const filterStatus = ref('')
    const dialogVisible = ref(false)
    const importDialogVisible = ref(false)
    const isEdit = ref(false)
    const formRef = ref(null)
    const currentId = ref(null)

    const form = reactive({
      name: '',
      email: '',
      phone: '',
      professionalField: '',
      title: '',
      organization: '',
      contactStatus: '待联系'
    })

    const rules = {
      name: [{ required: true, message: '请输入姓名', trigger: 'blur' }],
      email: [
        { required: true, message: '请输入邮箱', trigger: 'blur' },
        { type: 'email', message: '邮箱格式不正确', trigger: 'blur' }
      ]
    }

    const fetchExperts = async () => {
      loading.value = true
      try {
        const res = await expertApi.getExperts(currentPage.value - 1, pageSize.value)
        if (res.data.success) {
          experts.value = res.data.data
          total.value = res.data.totalElements
        }
      } catch (error) {
        ElMessage.error('获取专家列表失败')
      } finally {
        loading.value = false
      }
    }

    const handleFilter = async () => {
      if (!filterField.value && !filterStatus.value) {
        fetchExperts()
        return
      }
      loading.value = true
      try {
        const res = await expertApi.filterExperts({
          field: filterField.value,
          status: filterStatus.value
        })
        if (res.data.success) {
          experts.value = res.data.data
          total.value = res.data.data.length
        }
      } catch (error) {
        ElMessage.error('筛选失败')
      } finally {
        loading.value = false
      }
    }

    const resetFilter = () => {
      filterField.value = ''
      filterStatus.value = ''
      searchKeyword.value = ''
      fetchExperts()
    }

    const exportExperts = async () => {
      try {
        const res = await excelApi.exportExperts()
        const blob = new Blob([res.data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' })
        const link = document.createElement('a')
        link.href = URL.createObjectURL(blob)
        link.download = 'experts.xlsx'
        link.click()
        ElMessage.success('导出成功')
      } catch (error) {
        ElMessage.error('导出失败')
      }
    }

    const showImportDialog = () => {
      importDialogVisible.value = true
    }

    const handleImport = async (file) => {
      try {
        const res = await excelApi.importExperts(file.raw)
        if (res.data.success) {
          ElMessage.success(res.data.message)
          importDialogVisible.value = false
          fetchExperts()
        }
      } catch (error) {
        ElMessage.error(error.response?.data?.message || '导入失败')
      }
    }

    const downloadTemplate = async () => {
      try {
        const res = await excelApi.downloadTemplate()
        const blob = new Blob([res.data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' })
        const link = document.createElement('a')
        link.href = URL.createObjectURL(blob)
        link.download = 'expert_import_template.xlsx'
        link.click()
      } catch (error) {
        ElMessage.error('下载模板失败')
      }
    }

    const handleSearch = async () => {
      if (!searchKeyword.value) {
        fetchExperts()
        return
      }
      loading.value = true
      try {
        const res = await expertApi.searchExperts(searchKeyword.value)
        if (res.data.success) {
          experts.value = res.data.data
          total.value = res.data.data.length
        }
      } catch (error) {
        ElMessage.error('搜索失败')
      } finally {
        loading.value = false
      }
    }

    const showAddDialog = () => {
      isEdit.value = false
      Object.assign(form, {
        name: '',
        email: '',
        phone: '',
        professionalField: '',
        title: '',
        organization: '',
        contactStatus: '待联系'
      })
      dialogVisible.value = true
    }

    const editExpert = (row) => {
      isEdit.value = true
      currentId.value = row.id
      Object.assign(form, row)
      dialogVisible.value = true
    }

    const submitForm = async () => {
      const valid = await formRef.value.validate().catch(() => false)
      if (!valid) return

      try {
        if (isEdit.value) {
          await expertApi.updateExpert(currentId.value, form)
          ElMessage.success('更新成功')
        } else {
          await expertApi.createExpert(form)
          ElMessage.success('添加成功')
        }
        dialogVisible.value = false
        fetchExperts()
      } catch (error) {
        ElMessage.error(error.response?.data?.message || '操作失败')
      }
    }

    const deleteExpert = async (row) => {
      try {
        await ElMessageBox.confirm('确定删除该专家吗？', '提示', {
          type: 'warning'
        })
        await expertApi.deleteExpert(row.id)
        ElMessage.success('删除成功')
        fetchExperts()
      } catch (error) {
        if (error !== 'cancel') {
          ElMessage.error('删除失败')
        }
      }
    }

    const viewDetail = (row) => {
      router.push(`/expert/${row.id}`)
    }

    const getStatusType = (status) => {
      const types = {
        '待联系': 'info',
        '已联系': 'warning',
        '确认参加': 'success',
        '暂不参与': 'danger'
      }
      return types[status] || 'info'
    }

    const handleSizeChange = (val) => {
      pageSize.value = val
      fetchExperts()
    }

    const handleCurrentChange = (val) => {
      currentPage.value = val
      fetchExperts()
    }

    onMounted(fetchExperts)

    return {
      loading,
      experts,
      currentPage,
      pageSize,
      total,
      searchKeyword,
      filterField,
      filterStatus,
      dialogVisible,
      importDialogVisible,
      isEdit,
      form,
      formRef,
      rules,
      handleSearch,
      handleFilter,
      resetFilter,
      showAddDialog,
      editExpert,
      submitForm,
      deleteExpert,
      viewDetail,
      getStatusType,
      handleSizeChange,
      handleCurrentChange,
      exportExperts,
      showImportDialog,
      handleImport,
      downloadTemplate
    }
  }
}
</script>

<style scoped>
.expert-list {
  max-width: 1400px;
  margin: 0 auto;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.search-bar {
  margin-bottom: 20px;
}

.pagination {
  margin-top: 20px;
  justify-content: flex-end;
}
</style>
